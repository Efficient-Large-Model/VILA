import argparse
import cv2
import time
import tempfile
import os
import sys
from termcolor import colored

# Ensure we can find the llava module if running from the root of the repo
# This is classic python path hacking to make sure imports work without package installation
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import llava
from llava import conversation as clib
from llava.media import Image

def capture_frame(camera_index=0):
    """
    Opens the webcam, allows it to warm up, and grabs a single frame.
    Returns the frame (numpy array) or None if something went wrong.
    """
    print(f"[*] Connecting to webcam (ID: {camera_index})...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[!] Error: Could not open the webcam. check connection?")
        return None

    # Cameras often need a moment to adjust to light levels (auto-exposure)
    # 15 frames is usually enough to settle the sensor
    print("[*] Warming up camera sensor...")
    for _ in range(15):
        cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[!] Error: Failed to grab a frame from the camera.")
        return None
    
    return frame

def processing_loop(model, args):
    """
    Main application loop: Capture -> Save -> Inference -> Cleanup
    """
    # Set up the conversation template
    # 'auto' usually does a good job, but we default to vicuna if unsure
    conv_template = clib.conv_templates.get(args.conv_mode)
    if not conv_template and args.conv_mode != 'auto':
        print(f"[!] Warning: Conversation mode '{args.conv_mode}' not found, defaulting to 'vicuna_v1'")
        args.conv_mode = 'vicuna_v1'

    # We set the global conversation just in case, though we usually manage it per loop
    if args.conv_mode != 'auto':
        clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

    while True:
        try:
            # 1. Snap a picture
            print("\n" + "-"*40)
            print("Say cheese! 📸 Capturing image...")
            frame = capture_frame()
            
            if frame is None:
                print("[!] Aborting loop due to camera error.")
                break

            # VILA expects a file path for the image, so we save it temporarily.
            # cv2 writes images as BGR, but that's fine for standard file formats like jpg.
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_filename = tmp.name
                cv2.imwrite(temp_filename, frame)
            
            print(f"[*] Image saved to temporary file: {temp_filename}")

            # 2. Build the prompt
            prompt_parts = []
            
            # The model expects our custom Image object wrapper
            media_obj = Image(temp_filename)
            prompt_parts.append(media_obj)
            prompt_parts.append(args.text)

            # 3. Run Inference
            print("[*] Model is thinking...")
            start_time = time.time()
            
            # generate_content handles the tokenization and model forward pass
            response = model.generate_content(prompt_parts)
            
            end_time = time.time()
            duration = end_time - start_time

            # 4. Display result
            print("\n" + "="*10 + " VILA RESPONSE " + "="*10)
            print(colored(response, "green", attrs=["bold"]))
            print("="*35)
            print(f"[*] Inference took {duration:.2f} seconds")

            # Cleanup the temp file so we don't fill up the drive
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

            # Loop control
            if not args.run_loop:
                break
            
            print(f"\nWaiting {args.loop_delay} seconds before next capture... (Press Ctrl+C to stop)")
            time.sleep(args.loop_delay)

        except KeyboardInterrupt:
            print("\n[*] Stopping demo. Goodbye!")
            # Clean up usage if we exit mid-loop
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                os.remove(temp_filename)
            break
        except Exception as e:
            print(f"\n[!] Unexpected error: {e}")
            break

def main():
    parser = argparse.ArgumentParser(description="Run VILA inference live from your webcam.")
    
    parser.add_argument("--model-path", "-m", type=str, required=True, 
                        help="Path to the model directory (e.g. VILA1.5-3B)")
    parser.add_argument("--lora-path", "-l", type=str, default=None, 
                        help="Optional path to a LoRA checkpoint")
    parser.add_argument("--conv-mode", "-c", type=str, default="auto", 
                        help="Conversation mode/template (default: auto)")
    parser.add_argument("--text", type=str, default="Describe what you see in the image.", 
                        help="The prompt to send to the VILA model")
    parser.add_argument("--run-loop", action="store_true", 
                        help="Run continuously in a loop")
    parser.add_argument("--loop-delay", type=float, default=3.0, 
                        help="Wait time (seconds) between captures in loop mode")

    args = parser.parse_args()

    # Load the model
    print(f"[*] Loading model from: {args.model_path}")
    print("[*] This might take a minute, especially for larger models...")
    
    try:
        # Load the unified VILA model
        if args.lora_path:
            model = llava.load(args.lora_path, model_base=args.model_path)
        else:
            model = llava.load(args.model_path, model_base=None)
        print("[*] Model loaded successfully!")
    except Exception as e:
        print(f"[!] Failed to load model. Is the path correct?\nError: {e}")
        return

    # Start the application loop
    processing_loop(model, args)

if __name__ == "__main__":
    main()

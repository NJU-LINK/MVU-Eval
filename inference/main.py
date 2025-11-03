from decord import VideoReader, cpu
import copy
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from openai import OpenAI
import time
from tqdm import tqdm
from utils.logging_utils import setup_logging, log_arguments
from datetime import datetime
import argparse
import json
import logging


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode the content of a local file to base64 format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(file_path, "rb") as f:
            encoded_content = base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        raise IOError(f"Error reading or encoding file: {e}")
    return encoded_content


def encode_video(video_path, max_pixels: int, max_frames: int = 32):
    """
    Read a video and return uniformly-sampled frames encoded as base64 JPEG strings.
    Frames are resized to keep aspect ratio with max side = max_pixels.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    target_max_size = max_pixels
    if original_width > target_max_size or original_height > target_max_size:
        n_width = int(target_max_size * original_width / max(original_width, original_height))
        n_height = int(target_max_size * original_height / max(original_width, original_height))
    else:
        n_width = original_width
        n_height = original_height
    logging.info(f"Original: {original_width}x{original_height} | Resized: {n_width}x{n_height}")

    # Decord reader with target resolution
    vr = VideoReader(
        video_path,
        ctx=cpu(0),
        num_threads=4,
        width=n_width,
        height=n_height
    )

    total_frame_num = len(vr)
    if total_frame_num == 0:
        logging.warning(f"No frames decoded for: {video_path}")
        return []

    if total_frame_num > max_frames:
        frame_idx = np.linspace(0, total_frame_num - 1, max_frames, dtype=int).tolist()
    else:
        frame_idx = list(range(total_frame_num))

    frames = vr.get_batch(frame_idx).asnumpy()
    logging.info(f"Decoded frames: {len(frames)} / total: {total_frame_num}")

    base64_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        output_buffer = BytesIO()
        img.save(output_buffer, format="JPEG")
        base64_str = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
        base64_frames.append(base64_str)
    return base64_frames


def build_messages_base(data_root: str, QA: dict, videos_l: list, nframes: int, max_pixels: int):
    """Build multi-modal message content: text + multiple image frames per video."""
    logging.info(f"QA meta: index={QA.get('QA_index')} | task={QA.get('task')}")

    contents = []
    for vid_idx, file_name in enumerate(videos_l):
        file_path = os.path.join(data_root, file_name)
        frames_b64 = encode_video(file_path, max_pixels=max_pixels, max_frames=nframes)
        logging.info(f"[Video {vid_idx+1}] {file_name} -> {len(frames_b64)} frames")

        # Add a small header for each video
        contents.append({"type": "text", "text": f"[Video {vid_idx+1}] {file_name}"})
        # Append each frame as image_url (OpenAI/vLLM-compatible)
        for b64 in frames_b64:
            contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                },
            })

    question = QA['question']
    options = '\n'.join(QA['options'])
    contents.extend([
        {"type": "text", "text": question},
        {"type": "text", "text": "Please select the correct answer from the options. Answer with the option’s letter directly."},
        {"type": "text", "text": options},
    ])

    # Light logging (avoid printing base64)
    for c in contents:
        if c['type'] == 'text':
            logging.info(f"[TEXT] {c['text'][:200]}")
        elif c['type'] == 'image_url':
            pass  # omit base64 in logs

    return contents


def process_QA(client, data_root, QA, nframes, model_name, max_pixels):
    """Build messages and call the model once for this QA item."""
    try:
        messages = build_messages_base(data_root, QA, QA['video_paths'], nframes, max_pixels)
    except Exception as e:
        logging.error(f"Failed to build messages for QA_index={QA.get('QA_index')}: {e}")
        return None

    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": messages
            }],
        )
        model_output = chat_response.choices[0].message.content
        return model_output
    except Exception as e:
        logging.error(f"Model call failed (QA_index={QA.get('QA_index')}): {e}, {type(e)}")
        return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="MVU-Eval inference runner")
    parser.add_argument("--model_name", type=str, required=True, help="e.g., Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_filename", type=str, required=True, help="QA JSON filename under QA_output/")
    parser.add_argument("--port", type=int, required=True, help="OpenAI-compatible server port (vLLM)")
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--nframes", type=int, default=32, help="#frames per video")
    parser.add_argument("--max_pixels", type=int, default=720, help="max side (px) for resized frames")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing all videos")
    return parser.parse_args()


def load_qa_data(data_filename):
    data_path = f'QA_output/{data_filename}'
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(record, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)
        f.write('\n')


def main():
    args = parse_arguments()

    openai_api_key = "sk-abc123"
    openai_api_base = f"http://localhost:{args.port}/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    LOG_DIR = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_name = os.path.basename(__file__).split('.')[0]
    MODEL_NAME = args.model_name.replace("/", "_")
    outfile_name_suffix = f'{exp_name}_{MODEL_NAME}_{timestamp}'
    log_dir_path = os.path.join(LOG_DIR, 'logs', args.data_filename.split('.')[0])
    os.makedirs(log_dir_path, exist_ok=True)

    setup_logging(log_dir_path, outfile_name_suffix)
    logging.info("="*60 + " EXPERIMENT START " + "="*60)
    log_arguments(args)
    logging.info("Data root: %s", args.data_root)

    qa_data = load_qa_data(args.data_filename)

    out_dir_path = os.path.join(
        LOG_DIR,
        'Model_output',
        f'max_pixel_{args.max_pixels}_nframes_{args.nframes}',
        args.data_filename.split('.')[0],
        exp_name
    )
    os.makedirs(out_dir_path, exist_ok=True)
    output_file = os.path.join(out_dir_path, f'{exp_name}_{MODEL_NAME}.json')

    existing_indexes = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_records = [json.loads(line) for line in f]
        existing_indexes = [r.get('QA_index') for r in existing_records]
        logging.info(f"Already processed QA groups: {len(existing_indexes)}")
        print(f"Already processed QA groups: {len(existing_indexes)}")

    unresolved_QAs = [QA for QA in qa_data if QA.get('QA_index') not in existing_indexes]
    logging.info(f"Unresolved QA groups: {len(unresolved_QAs)}")
    print(f"Unresolved QA groups: {len(unresolved_QAs)}")

    for QA in tqdm(unresolved_QAs, desc=f"Processing {MODEL_NAME}"):
        model_output = process_QA(
            client=client,
            data_root=args.data_root,
            QA=QA,
            nframes=args.nframes,
            model_name=args.model_name,
            max_pixels=args.max_pixels
        )
        if model_output is None:
            continue

        # assemble and save record
        record = copy.deepcopy(QA)
        if 'model_results' not in record:
            record['model_results'] = {}
        record['model_results'][MODEL_NAME] = {'model_output': model_output}
        save_results(record, output_file)

    logging.info("="*60 + " EXPERIMENT DONE " + "="*60)


if __name__ == "__main__":
    main()

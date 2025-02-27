import os
import json
import random
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API")
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


def search_youtube_video(video_title, cache):
    if video_title in cache:
        print('hit cache')
        return cache[video_title],True

    params = {
        "part": "snippet",
        "q": video_title,
        "type": "video",
        "key": API_KEY,
        "maxResults": 1
    }
    response = requests.get(SEARCH_URL, params=params)
    print(response.json())
    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            video_id = data["items"][0]["id"]["videoId"]
            video_link = f"https://www.youtube.com/watch?v={video_id}"
            cache[video_title] = video_link
            return video_link,False
    return "",False


def update_json_file(file_path, output_file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cache = {}
    for i, item in enumerate(data):
        if not item.get("video_link"):
            print(f"{i} Searching for: {item['video_name']}")
            item["video_link"],hit = search_youtube_video(item["video_name"], cache)
            tmp = item["video_link"]
            print(f"result = {tmp}")
            if not hit:
                time.sleep(random.randint(0,100)/100)

            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    json_file_path = "./faiss/faiss_hnsw_index_metadata.json"
    # output_file_path = "./faiss/updated_faiss_index_metadata.json"
    output_file_path = "./faiss/faiss_hnsw_index_metadata_output.json"
    if os.path.exists(json_file_path):
        update_json_file(json_file_path, output_file_path)
        print("JSON file updated successfully.")
    else:
        print("JSON file not found.")

import json


def extract_video_links(input_file, output_file):
    # Read JSON data
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extract relevant information and remove duplicates
    seen_titles = set()
    extracted_data = []
    for item in data:
        title = item["video_name"]
        if title not in seen_titles:
            seen_titles.add(title)
            extracted_data.append({
                "title": title,
                "url": item["video_link"]
            })

    # Write to new JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(extracted_data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file = "./faiss/faiss_hnsw_index_metadata.json"
    output_file = "./configs/video_url.json"
    extract_video_links(input_file, output_file)
import pandas as pd
import json
from collections import Counter

# Read the filtered CSV file
df = pd.read_csv('data/filtered_leetcode_dataset.csv')

# Initialize a list to store all topics
all_topics = []

# Process the related_topics column
for topics_str in df['related_topics'].dropna():
    # Split topics by comma and strip whitespace
    topics = [topic.strip() for topic in str(topics_str).split(',')]
    all_topics.extend(topics)

# Count unique topics
topic_counter = Counter(all_topics)

# Create result dictionary
result = {
    "total_unique_topics": len(topic_counter),
    "topics": {
        topic: count for topic, count in sorted(topic_counter.items(), key=lambda x: (-x[1], x[0]))
    }
}

# Save to JSON file
with open('topic_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"Total number of unique topics: {len(topic_counter)}")
print("Results have been saved to topic_analysis.json") 
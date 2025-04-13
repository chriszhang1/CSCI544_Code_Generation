import pandas as pd
import json
from collections import Counter

# Read the current topics from topic_analysis.json
with open('topic_analysis.json', 'r') as f:
    topic_data = json.load(f)
    allowed_topics = set(topic_data['topics'].keys())

# Read the original dataset
df = pd.read_csv('data/leetcode_dataset - lc.csv')

# Function to check if at least one topic is in allowed_topics
def has_any_allowed_topics(topics_str):
    if pd.isna(topics_str):
        return False
    topics = [topic.strip() for topic in str(topics_str).split(',')]
    return any(topic in allowed_topics for topic in topics)

# Function to clean topics - keep only allowed topics
def clean_topics(topics_str):
    if pd.isna(topics_str):
        return ''
    topics = [topic.strip() for topic in str(topics_str).split(',')]
    allowed_only = [topic for topic in topics if topic in allowed_topics]
    return ', '.join(allowed_only)

# Filter the dataset - relaxed mode (at least one topic must be allowed)
filtered_df = df[df['related_topics'].apply(has_any_allowed_topics)]

# Clean up the topics - remove non-allowed topics
filtered_df['related_topics'] = filtered_df['related_topics'].apply(clean_topics)

# Save the filtered dataset
filtered_df.to_csv('data/filtered_leetcode_dataset.csv', index=False)

print(f"Original dataset size: {len(df)}")
print(f"Filtered dataset size (at least one allowed topic): {len(filtered_df)}")
print("Dataset has been saved to data/filtered_leetcode_dataset.csv")

# Update topic analysis for the cleaned dataset
all_topics = []
for topics_str in filtered_df['related_topics'].dropna():
    topics = [topic.strip() for topic in str(topics_str).split(',')]
    all_topics.extend(topics)

topic_counter = Counter(all_topics)
result = {
    "total_unique_topics": len(topic_counter),
    "topics": {
        topic: count for topic, count in sorted(topic_counter.items(), key=lambda x: (-x[1], x[0]))
    }
}

with open('topic_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nTopic analysis has been updated in topic_analysis.json") 
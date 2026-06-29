import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def generate_mock_scouting_data():
    """
    Generates a realistic scouting dataset representing 100 global players.
    Includes SC Braga's Ricardo Horta as the base profile.
    """
    np.random.seed(42)
    
    # Core performance metrics relevant to an attacking midfielder / winger like Horta
    metrics = [
        "goals_per_90", "assists_per_90", "shots_per_90", 
        "key_passes_per_90", 
        "expected_goals_xG_per_90", "pressing_duels_won_per_90"
    ]
    
    # Generate 99 random scouting targets
    data = np.random.uniform(0.1, 4.0, size=(99, len(metrics)))
    player_names = [f"Scouted Player {i+1}" for i in range(99)]
    df = pd.DataFrame(data, columns=metrics)
    df.insert(0, "player_name", player_names)
    df.insert(1, "current_club", np.random.choice(["League One", "Brasileirão", "Allsvenskan", "Eredivisie"], 99))
    
    # Manually append Ricardo Horta's profile for the algorithm to match against (source: fotmob/fbref)
    horta_profile = {
        "player_name": "Ricardo Horta",
        "current_club": "SC Braga",
        "goals_per_90": 0.59,
        "assists_per_90": 0.17,
        "shots_per_90": 2.55,
        "key_passes_per_90": 2.05,
        "expected_goals_xG_per_90": 0.51,
        "pressing_duels_won_per_90": 2.55
    }
    
    df = pd.concat([df, pd.DataFrame([horta_profile])], ignore_index=True)
    return df

def find_similar_players(df, target_player_name, top_n=5):
    """
    Standardizes the performance features and computes Cosine Similarity.
    """
    # Isolate num features for the ML model
    feature_columns = [col for col in df.columns if col not in ["player_name", "current_club"]]
    
    # This prevents features with naturally higher scales from dominating the model
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])
    
    similarity_matrix = cosine_similarity(scaled_features)
    
    # Map back to find the target player's row index
    try:
        target_idx = df[df["player_name"] == target_player_name].index[0]
    except IndexError:
        raise ValueError(f"Player '{target_player_name}' not found in the database.")
    
    # Get similarity scores for our target player against all others
    similarity_scores = list(enumerate(similarity_matrix[target_idx]))
    
    # Sort players based on highest similarity score (excluding themselves)
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    filtered_scores = [item for item in sorted_scores if item[0] != target_idx]
    
    # Extract top N recommendations
    recommendations = []
    for idx, score in filtered_scores[:top_n]:
        recommendations.append({
            "Rank": len(recommendations) + 1,
            "Name": df.iloc[idx]["player_name"],
            "Club": df.iloc[idx]["current_club"],
            "Similarity Score": round(score * 100, 2)  # Convert to percentage
        })
        
    return pd.DataFrame(recommendations)

if __name__ == "__main__":
    print("--- Executing SC Braga Scouting Pipeline ---")
    
    # Run pipeline
    raw_data = generate_mock_scouting_data()
    print(f" Successfully loaded dataset containing {len(raw_data)} player profiles.")
    
    print("\n[Processing] Running Cosine Similarity algorithms on Horta's matrix...")
    results = find_similar_players(raw_data, target_player_name="Ricardo Horta", top_n=10)
    
    print("\n--- TOP 10 TRANSFER TARGETS IDENTIFIED ---")
    print(results.to_string(index=False))
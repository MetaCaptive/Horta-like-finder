from fastapi import FastAPI, HTTPException
from app.scouting_engine import generate_mock_scouting_data, find_similar_players

app = FastAPI(title="SC Braga Recruitment API", version="1.0")


PLAYER_DATABASE = generate_mock_scouting_data()


@app.get("/scout/similarity")
def get_similar_players(player_name: str = "Ricardo Horta", limit: int = 10):
    """
    Exposes the Machine Learning model as a production-ready API endpoint.
    """
    try:
        # Run your ML pipeline inside the API request lifecycle
        recommendations = find_similar_players(PLAYER_DATABASE, target_player_name=player_name, top_n=limit)
        
        # Convert the pandas dataframe into a clean JSON response for the frontend
        return {
            "target_player": player_name,
            "status": "success",
            "results": recommendations.to_dict(orient="records")
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

####uvicorn app.main:app --reload

####No browser colocar: http://127.0.0.1:8000/docs
#### Para direct query: http://127.0.0.1:8000/scout/similarity?player_name=Ricardo Horta&limit=5 (limit a definir)
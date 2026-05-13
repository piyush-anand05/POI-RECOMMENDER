import requests

def get_place_name(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        res = requests.get(url, headers={"User-Agent": "poi-app"})
        data = res.json()
        return data.get("display_name", "Unknown Location")
    except:
        return "Unknown Location"

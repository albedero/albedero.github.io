"""
Bike Sharing Model Web Application
"""

from flask import Flask, request
import sklearn
from library.src import ie_bike_model
from library.src.ie_bike_model import model
import sys
import datetime


app = Flask(__name__)
print(__name__)


# app reciveves fun as param
@app.route("/")
def versions():
    version_dict = {
        "sklearn": sklearn.__version__,
        "python": sys.version,
        "ie-bike-model": ie_bike_model.__version__,
    }
    return version_dict


@app.route("/train_and_persist")
def do_train():
    evaluation_dict = model.train_and_persist()
    return {"status": "OK", "evaluation metrics": evaluation_dict}


@app.route("/predict")
def do_predict():
    now = datetime.datetime.now()

    weather_dict = {
        "clear": "Clear, Few clouds, Partly cloudy, Partly cloudy",
        "cloudy": "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist",
        "light_rain": "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",
        "heavy_rain": "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog",
    }

    result = model.predict(
        dteday=request.args["date"],
        hr=request.args["hour"],
        weathersit=weather_dict[request.args["weather_situation"]],
        temp=request.args["temperature"],
        atemp=request.args["feeling_temperature"],
        hum=request.args["humidity"],
        windspeed=request.args["windspeed"],
    )

    then = datetime.datetime.now()
    difference = then - now
    diff = difference.total_seconds()
    seconds = round(diff, 4)
    return {"result": int(result), "elapsed_time": seconds}


if __name__ == "__main__":
    print("success")
    print(sys.argv)
    port = 5435
    try:
        port = int(sys.argv[1])
    except IndexError:
        print("Please specify port")
    except ValueError:
        print("Please specify a proper integer port")
    else:
        app.run(host="0.0.0.0", port=port)

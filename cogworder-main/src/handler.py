import time
import subprocess
import os
import traceback
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry

TIMEOUT = int(os.environ.get("RUNPOD_REQUEST_TIMEOUT", "600"))

LOCAL_URL = "http://127.0.0.1:5000"

cog_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
cog_session.mount('http://', HTTPAdapter(max_retries=retries))


# ----------------------------- Start API Service ---------------------------- #
# Call "python -m cog.server.http" in a subprocess to start the API service.
subprocess.Popen(["python", "-m", "cog.server.http"])


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    '''
    Check if the service is ready to receive requests.
    '''
    while True:
        try:
            health = requests.get(url, timeout=120)
            status = health.json()["status"]

            if status == "READY":
                time.sleep(1)
                return

        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)


def run_inference(inference_request):
    '''
    Run inference on a request.
    '''
    print(f"[Run Interface] Got {inference_request=}")

    response = cog_session.post(url=f'{LOCAL_URL}/predictions',
                                json=inference_request, timeout=TIMEOUT)

    if response.status_code != 200:
        print("Request failed - reason :", response.status_code, response.text)

    return response.json()


def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''

    print(f"[Handler] Got {event=}")

    while True:
        try:
            json = run_inference(event["input"])
            return json["output"]
        except Exception as e:
            print(f"Got {e=}")
            traceback.print_exception(e)


if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/health-check')

    print("Cog API Service is ready. Starting RunPod serverless handler...")

    runpod.serverless.start({"handler": handler})

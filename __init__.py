import logging
import base64
import json
from PIL import Image
import io
import azure.functions as func
import datetime
# from memory_profiler import profile
 
from . import predict
from . import extract
# @profile
def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    timestamp = datetime.datetime.now()
    file_value = req.params.get('file')
    if not file_value:
        try:
            payload = req.get_json()
        except ValueError:
            pass
        else:
            file_value = payload.get('file')
            if file_value is None:
                return func.HttpResponse("File input is required")
            if 'settings' not in payload:
                payload['settings'] ={'crops':False}
            if 'crops' not in payload['settings']:
                payload['settings'] ={'crops':False}
            crop_settings = payload['settings']['crops']
            crop_settings = str(crop_settings)
    
    if 'base64' in file_value:
        base64str = file_value.split('base64,')[1]
        byte_data = base64.b64decode(base64str)
    else:
        img_buf=json.loads(file_value)
        byte_data = bytes([img_buf[key] for key in sorted(img_buf.keys(), key=int)])

    image_object = Image.open(io.BytesIO(byte_data))
    detections = predict.predict_img(image_object)
    response_data = []
   
    for detection in detections:
        if len(detection.boxes) == 0:
            count = 0
            # response_data=[count]
            # print(f"signature not found")
        else:
            count = len(detection.boxes)
            box=detection.boxes[0]
            for box in detection.boxes:
                cords = box.xyxy[0].tolist()
                rcords = [round(x) for x in cords]
                class_id = detection.names[box.cls[0].item()]
                conf = round(box.conf[0].item(), 2)
                boundingRegions = {"polygon": [
                                        {"x": rcords[0], "y": rcords[1]},
                                        {"x": rcords[0], "y": rcords[3]},
                                        {"x": rcords[2], "y": rcords[3]},
                                        {"x": rcords[2], "y": rcords[1]}]}
                #for base64 of crops
                if "true" in crop_settings or "True" in crop_settings:                     
                    base64_image = extract.save_crop(image_object, cords)
                    base64_str = "data:image/jpeg;base64," + base64_image
                    response_data.append({"type": class_id, "confidence": conf,  "bounding_regions": boundingRegions, "content": base64_str})
                    # response_data[].append({"type": class_id, "probability": conf, "Signature": base64_image})
                else:
                    # response_data.append({"Object type": class_id, "Coordinates": rcords, "Probability": conf})
                    response_data.append({"type": class_id, "confidence": conf,  "bounding_regions": boundingRegions})

    response = {
    "context": {
        "timestamp": timestamp.isoformat(),
        "transaction_id": context.invocation_id
    },
    "signatures" : count,
    "result": response_data
    }
    
    return func.HttpResponse(json.dumps(response), status_code=200)

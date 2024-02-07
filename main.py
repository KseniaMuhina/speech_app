import whisper
import time


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Time {func.__name__}: {execution_time / 60}")
        return result
    return wrapper


def load_model(type_of_model: str):
    return whisper.load_model(type_of_model)


@timing_decorator
def transcribe_meeting(model):
    result = model.transcribe("/home/ksenia/PycharmProjects/whisper_app/rt_podcast889.mp3", fp16=False)
    with open('transcription.txt', 'w') as f:
        f.write(result['text'])


model = load_model('base')
transcribe_meeting(model)

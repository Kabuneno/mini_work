from datasets import load_dataset, concatenate_datasets, Dataset, Audio, DatasetDict
from scipy.io.wavfile import write

import soundfile as sf
import numpy as np
from tqdm import tqdm

import os
import torch

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Trainer, TrainingArguments
from dataclasses import dataclass
from typing import Dict, List, Union, Any
import os
import torch
from datasets import load_dataset, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Trainer, TrainingArguments

from typing import Any, Dict, List, Union

# В

ds = load_dataset("Simonlob/Kany_dataset_mk4_Base")
dataset = concatenate_datasets([ds['train'],ds['test']])
# print(enumerate(dataset))
# for i,ax in tqdm(enumerate(dataset),total=2,desc="Сохранение аудиофайлов"):
#   print(i,ax)
hui = dataset.select(range(2))

save_path = "audio_dataset"
os.makedirs(save_path, exist_ok=True)

# for i, example in tqdm(enumerate(dataset), total=2, desc="Сохранение аудиофайлов"):
#     audio_array = np.array(example["audio"]["array"], dtype = np.float32)
#     sampling_rate = example["audio"]["sampling_rate"]
#     file_name = os.path.join(save_path, f"{i}.wav")
#     write(file_name, sampling_rate, audio_array)

    # with sf.SoundFile(file_name, mode='w', samplerate=sampling_rate, channels=1, subtype='FLOAT', ) as file:
    #     file.write(audio_array)
for i, example in tqdm(enumerate(hui), desc="Сохранение аудиофайлов"):
    audio_array = np.array(example["audio"]["array"], dtype=np.float32)
    sampling_rate = example["audio"]["sampling_rate"]
    file_name = os.path.join(save_path, f"{i}.wav")
    write(file_name, sampling_rate, audio_array)

print(f" Все {2} аудиофайлв сохранены в папке {save_path}")





model_name = "the-cramer-project/AkylAI-STT-small"
processor = AutoProcessor.from_pretrained(model_name)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")



from evaluate import load
import jiwer

def safe_map(dataset, preprocess_fn):
    def apply_preprocess(example):
        try:
            return preprocess_fn(example)
        except Exception as e:
            print(f"Ошибка при обработке примера: {e}")
            return None

    # processed = dataset.map(apply_preprocess, remove_columns=dataset.column_names)
    processed = dataset.map(apply_preprocess)

    return processed.filter(lambda x: x is not None)

# def prepare_dataset(batch):
#     if "audio" not in batch:
#         print(f"Ошибка: отсутствует ключ 'audio' в {batch.keys()}")
#     return None

#     audio = batch.get("audio", None)
#     transcription = batch.get("transcription", "")

#     if audio is None:
#         print("Предупреждение: отсутствует аудио")
#         return None

#     import io
#     import soundfile as sf
#     from scipy import signal

#     if isinstance(audio, dict) and "bytes" in audio:
#         audio_data, sample_rate = sf.read(io.BytesIO(audio["bytes"]))
#     elif isinstance(audio, dict) and "array" in audio:
#         audio_data = audio["array"]
#         sample_rate = audio.get("sampling_rate", 16000)
#     else:
#         print(f"Неподдерживаемый формат аудио: {type(audio)}")
#         return None

#     if sample_rate != 16000:
#         audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))

#     return {
#         "input_features": processor(
#             audio_data,
#             sampling_rate=16000,
#             return_tensors="pt"
#         ).input_features.squeeze(0),
#         "labels": processor.tokenizer(transcription).input_ids
#     }
def prepare_dataset(batch):
    if "audio" not in batch:
        print(f"Ошибка: отсутствует ключ 'audio' в {batch.keys()}")
        return None  # Только если ключа "audio" действительно нет!

    audio = batch.get("audio", None)
    transcription = batch.get("transcription", "")

    if audio is None:
        print("Предупреждение: отсутствует аудио")
        return None

    import io
    import soundfile as sf
    from scipy import signal

    if isinstance(audio, dict) and "bytes" in audio:
        audio_data, sample_rate = sf.read(io.BytesIO(audio["bytes"]))
    elif isinstance(audio, dict) and "array" in audio:
        audio_data = audio["array"]
        sample_rate = audio.get("sampling_rate", 16000)
    else:
        print(f"Неподдерживаемый формат аудио: {type(audio)}")
        return None

    if sample_rate != 16000:
        audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))

    return {
        "input_features": processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.squeeze(0).tolist(),  # Сохранение в list для совместимости с Dataset

        "labels": processor.tokenizer(transcription).input_ids
    }


kyrgyz_dataset = load_dataset("Simonlob/Kany_dataset_mk4_Base")
kyrgyz_dataset['train'] = kyrgyz_dataset['train'].select(range(15))

processed_dataset = {}
processed_dataset["train"] = safe_map(kyrgyz_dataset['train'], prepare_dataset)
splits = processed_dataset["train"].train_test_split(test_size=0.1)
processed_dataset["train"] = splits["train"]
processed_dataset["validation"] = splits["test"]

model_name = "the-cramer-project/AkylAI-STT-small"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)

special_kyrgyz_tokens = ["ң", "ү", "ө", "Ң", "Ү", "Ө"]
for token in special_kyrgyz_tokens:
    if token not in processor.tokenizer.get_vocab():
        # print(f"Добавление специального токена: {token}")
        processor.tokenizer.add_tokens(token)
        model.resize_token_embeddings(len(processor.tokenizer))

@dataclass

# class DataCollatorCTCWithPadding:
#     processor: Any
#     padding: Union[bool, str] = True
#     max_length: Union[int, None] = None
#     max_length_labels: Union[int, None] = None

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         input_features = [{"input_features": feature["input_features"]} for feature in features]
#         label_features = [{"input_ids": feature["labels"]} for feature in features]

#         batch = {
#             "input_features": torch.tensor([f["input_features"] for f in input_features])
#         }


#         # with self.processor.tokenizer.as_target_processor():
#         #     labels_batch = self.processor.tokenizer.pad(
#         #         label_features,
#         #         padding=self.padding,
#         #         max_length=self.max_length_labels,
#         #         return_tensors="pt"
#         #     )
#         with self.processor.tokenizer:
#            labels_batch = self.processor.tokenizer.pad(
#               label_features,
#               padding=self.padding,
#               return_tensors="pt",
# )




#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

#         batch["labels"] = labels
#         return batch

class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True
    max_length: Union[int, None] = None
    max_length_labels: Union[int, None] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = {
            "input_features": torch.tensor([f["input_features"] for f in input_features])
        }

        # Remove the with statement as it's not needed for padding
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

# class DataCollatorCTCWithPadding:
#     processor: Any
#     padding: Union[bool, str] = True
#     max_length: Union[int, None] = None
#     max_length_labels: Union[int, None] = None

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         input_features = [{"input_features": feature["input_features"]} for feature in features]
#         label_features = [{"input_ids": feature["labels"]} for feature in features]

#         batch = self.processor.pad(
#             input_features,
#             padding=self.padding,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )

#         with self.processor.tokenizer.as_target_processor():
#             labels_batch = self.processor.pad(
#                 label_features,
#                 padding=self.padding,
#                 max_length=self.max_length_labels,
#                 return_tensors="pt"
#             )

#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

#         batch["labels"] = labels
#         return batch

data_collator = DataCollatorCTCWithPadding(
    processor=processor,
    padding=True
)

# class LossLoggingCallback:
#     def __init__(self):
#         self.losses = []

#     def __call__(self, args, state, control, logs=None, **kwargs):
#         if state.is_local_process_zero and "loss" in logs:
#             print(f"Шаг {state.global_step}: Loss = {logs['loss']:.4f}")
#             self.losses.append(logs['loss'])

# class LossLoggingCallback:
#     def __init__(self):
#         self.losses = []

#     def on_log(self, args, state, control, logs=None, **kwargs): # Changed __call__ to on_log
#         if state.is_local_process_zero and "loss" in logs:
#             print(f"Шаг {state.global_step}: Loss = {logs['loss']:.4f}")
#             self.losses.append(logs['loss'])

#     def on_init_end(self, args, state, control, **kwargs): # Added on_init_end method
#         pass # or any initialization you want to do

from transformers import TrainerCallback

class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_train_begin(self, args, state, control, **kwargs):
        """Вызывается в начале тренировки"""
        print("Обучение началось!")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Вызывается при логировании данных"""
        if state.is_local_process_zero and "loss" in logs:
            print(f"Шаг {state.global_step}: Loss = {logs['loss']:.4f}")
            self.losses.append(logs['loss'])

wer_metric = load("wer")
import torch

def compute_metrics(pred):
    pred_logits = pred.predictions

    # pred_logits = torch.tensor(pred_logits)
    # pred_logits = torch.from_numpy(pred_logits)
    pred_logits = torch.from_numpy(pred.predictions[0]) if isinstance(pred.predictions, tuple) else torch.from_numpy(pred.predictions)

    print(pred_logits.shape)

    if pred_logits.ndim == 3:
          pred_logits = pred_logits.argmax(dim=-1)
    else:
          pred_logits = pred_logits.squeeze().argmax(dim=-1)



    if len(pred_logits.shape) > 2:
        pred_logits = pred_logits[:, 0, :]

    pred_ids = np.argmax(pred_logits.numpy(), axis=-1)
    pred_str = processor.batch_decode(pred_ids)

    label_ids = pred.label_ids
    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
    label_str = processor.batch_decode(label_ids)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}



# def compute_metrics(pred):
#     pred_logits = pred.predictions
#     pred_ids = np.argmax(pred_logits, axis=-1)

#     pred_str = processor.batch_decode(pred_ids)
#     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

#     wer = wer_metric.compute(predictions=pred_str, references=label_str)

#     return {"wer": wer}

training_args = TrainingArguments(
    output_dir="./akylai-kyrgyz",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=10,
    learning_rate=1e-4,
    warmup_steps=100,
    save_total_limit=2,
    fp16=True if device == "cuda" else False,
    push_to_hub=False,
    remove_unused_columns=False
)

loss_callback = LossLoggingCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[loss_callback]
)

torch.cuda.empty_cache()

print("Начало обучения...")
train_result = trainer.train()

trainer.save_model("./akylai-kyrgyz-final")
processor.save_pretrained("./akylai-kyrgyz-final")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(loss_callback.losses)
plt.title('Динамика Loss во время обучения')
plt.xlabel('Шаг')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('loss_plot.png')
plt.close()



def transcribe_audio(audio_path):
    import soundfile as sf
    from scipy import signal
    # import torchaudio
    # audio_input, sample_rate = torchaudio.load(audio_path)

    audio_input, sample_rate = sf.read(audio_path,format="WAV")

    if sample_rate != 16000:
        audio_input = signal.resample(audio_input, int(len(audio_input) * 16000 / sample_rate))

    inputs = processor(
        audio_input,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(inputs.input_features.to(device)).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription

# def test_model(model, processor, test_dataset):
#     print("\nТестирование модели:")
#     for i in range(min(5, len(test_dataset))):
#         import tempfile
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#             temp_audio_path = f.name
#             print(f"Размер файла: {os.path.getsize(temp_audio_path)} байт")
#             sample = test_dataset[i]
#             audio_data = sample["audio"]

#             if isinstance(audio_data, dict) and "bytes" in audio_data:
#                 f.write(audio_data["bytes"])

#             transcription = transcribe_audio(temp_audio_path)
#             os.remove(temp_audio_path)

#         print(f"\nПример {i+1}:")
#         print(f"Оригинальная транскрипция: {sample.get('transcription', 'Н/Д')}")
#         print(f"Предсказанная транскрипция: {transcription}")
print(processed_dataset, "НАШЩАВААФААААА АПАРАШААВАПУП")

def test_model(model, processor, test_dataset):
    print("\nТестирование модели:")
    for i in range(min(5, len(test_dataset))):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_audio_path = f.name
            sample = test_dataset[i]
            audio_data = sample["audio"]['array']
            print(audio_data)
            print(sample)
            print(temp_audio_path, " FIND THISSADFFDWD")
            if isinstance(audio_data, dict) and "bytes" in audio_data:
                if len(audio_data["bytes"]) == 0:
                    print(f"Ошибка: пустые аудиоданные в примере {i+1}")
                    continue  # Пропускаем этот пример

                f.write(audio_data["bytes"])
                f.flush()  # Убедимся, что данные записаны

            else:
                print(f"Ошибка: отсутствуют корректные аудиоданные в примере {i+1}")
                continue

        transcription = transcribe_audio(temp_audio_path)
        os.remove(temp_audio_path)

        print(f"\nПример {i+1}:")
        print(f"Оригинальная транскрипция: {sample.get('transcription', 'Н/Д')}")
        print(f"Предсказанная транскрипция: {transcription}")


test_model(model, processor, processed_dataset["validation"])

print("Обучение и тестирование завершено!")








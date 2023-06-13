import evaluate
from comet import download_model, load_from_checkpoint

predictions = [
    "This is an examplary sentence.",
    "Try different sentences for the reference and observe the change in scores.",
]
references = [
    ["This is an examplary sentence."],
    ["Try different sentences for the reference and observe the change in scores."],
]

comet = evaluate.load("comet")
comet_score = comet.compute(
    predictions=hypothesis, references=reference, sources=source
)
print(round(results["score"], 1))

print("Notice the comet-qe-da metric does not use references!")
print("See Section 8 https://aclanthology.org/2020.wmt-1.101/")
qe_model_path = download_model("wmt20-comet-qe-da")
comet_qe_da = load_from_checkpoint(qe_model_path)
data_no_ref = [{"src": src, "mt": mt} for src, mt in zip(source, hypothesis)]
qe_comet_score = comet_qe_da.predict(data_no_ref)
print(qe_comet_score)

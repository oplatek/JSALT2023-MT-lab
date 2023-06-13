import evaluate
from comet import download_model, load_from_checkpoint

source = [
    "Dem Feuer konnte Einhalt geboten werden",
    "Schulen und Kindergärten wurden eröffnet.",
]
hypothesis = [
    "The fire could be stopped",
    "Schools and kindergartens were open",
]
reference = [
    "They were able to control the fire.",
    "Schools and kindergartens opened",
]

comet = evaluate.load("comet")
comet_score = comet.compute(
    predictions=hypothesis, references=reference, sources=source
)
print(comet_score)

print("\n\nNotice the comet-qe-da metric does not use references!")
print("See Section 8 https://aclanthology.org/2020.wmt-1.101/")
qe_model_path = download_model("wmt20-comet-qe-da")
comet_qe_da = load_from_checkpoint(qe_model_path)
data_no_ref = [{"src": src, "mt": mt} for src, mt in zip(source, hypothesis)]
qe_comet_score = comet_qe_da.predict(data_no_ref)
print(qe_comet_score)

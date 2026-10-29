import nltk


def download_resources():
    resources = [
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger_eng",
    ]
    for resource in resources:
        print(f"Downloading NLTK resource: {resource}...")
        nltk.download(resource, quiet=False)
    print("All required NLTK resources downloaded successfully.")


if __name__ == "__main__":
    download_resources()

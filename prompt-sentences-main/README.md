# prompt-sentences

**Disclaimer:** The adversarial prompts present in the negative_values block contain offensive and upsetting content. Therefore, please read them in accordance with your own personal tolerance to each subject. Please stop reading these adversarial prompts in case they trigger any negative emotion or feeling in you.

## Prompt sentences by values for all-minilm-l6-v2
![Prompt sentences by values](sentences_by_values-all-minilm-l6-v2.png)

## Prompt sentences by values for bge-large-en-v1.5
![Prompt sentences by values](sentences_by_values-bge-large-en-v1.5.png)

## Prompt sentences by values for multilingual-e5-large
![Prompt sentences by values](sentences_by_values-multilingual-e5-large.png)

## Data structure

Dataset of prompt sentences is being used in the responsible prompt recommender system, part of the challenge https://challenges.apps.res.ibm.com/challenges/6550

The dataset is in json format and is organized in two blocks, i.e., positive values and negative values. Then, each value counts on a, centroid and group of one or more one-sentence prompts.

    {
    "positive_values":[
        {
            "label": "...",
            "prompts": [
                { "text": "...", "ref": ..., "embedding": [] }
                { "text": "...", "ref": ..., "embedding": [] }
                { "text": "...", "ref": ..., "embedding": [] }
                ...
            ],
            "centroid": []
        },
        {
            "label": "...",
            "prompts": [
                { "text": "...", "ref": ..., "embedding": [] }
                { "text": "...", "ref": ..., "embedding": [] }
                { "text": "...", "ref": ..., "embedding": [] }
                ...
            ],
            "centroid": []
        },
        ...
    ],
    "negative_values": [
        {
            "label": "...",
            "prompts": [
                { "text": "...", "ref": ..., "embedding": [] }
                { "text": "...", "ref": ..., "embedding": [] }
                { "text": "...", "ref": ..., "embedding": [] }
                ...
            ],
            "centroid": []
        },
        {
            "label": "...",
            "prompts": [
                { "text": "...", "ref": ..., "embedding": [] }
                { "text": "...", "ref": ..., "embedding": [] }
                { "text": "...", "ref": ..., "embedding": [] }
                ...
            ],
            "centroid": []
        },
        ...
    ]
    }


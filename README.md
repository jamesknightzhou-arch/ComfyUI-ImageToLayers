# ComfyUI-ImageToLayers

Separate any image (including AI-generated ones) into **editable layers** inside ComfyUI, using
[Image to Layers](https://www.imagetolayers.com) "Detailed Layers": the AI finds the editable elements, returns each as its own
transparent layer, and rebuilds the background behind them.

## Nodes

| Node | What it does |
| --- | --- |
| **Image to Layers (Detailed)** | Sends the image, waits for the job, returns the layers bottom to top. Optional `parts_to_separate` ("bottle, leaves, podium"). |
| **Image to Layers: Pick Layer** | Takes layer `index` out of the batch (0 is the background). |
| **Image to Layers: Account** | Shows who is signed in and the credits left; can sign out. |

Outputs of *Detailed*:

- `layers` – `IMAGE` batch, one image per layer, all the size of the background layer (the layers sit where they were in the picture).
- `masks` – `MASK` batch, the alpha of each layer (**1 = the layer's pixels**; ComfyUI's `JoinImageWithAlpha` expects the inverse for
  `LoadImage` masks, so use *Pick Layer* → `JoinImageWithAlpha` as in the example workflow to get a transparent PNG).
- `layer_names` – JSON list of the layer names, same order.
- `job_id` – the job on imagetolayers.com.

## Install

ComfyUI Manager → search **Image to Layers**, or:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jamesknightzhou-arch/ComfyUI-ImageToLayers
pip install -r ComfyUI-ImageToLayers/requirements.txt
```

Restart ComfyUI.

## Sign in (no API key)

The first time you run the node it stops with a message containing a link and a short code. Open the link, check the code, approve
(email or Google, in your browser), then queue the workflow again. The node picks the approval up by itself.

The session is stored on the machine that runs ComfyUI in `~/.imagetolayers/comfyui_session.json` (mode 600) and refreshes itself.
Use **Image to Layers: Account** with *Sign out* to remove it. No password or key is ever stored.

## Credits and privacy

- Each run uses credits from your Image to Layers account, the same as Detailed Layers on the website.
- The image you connect is uploaded to imagetolayers.com for processing. Do not connect images you do not have the right to process.

## Example

`workflow_image_to_layers.json` loads an image, separates it, and saves layer 1 as a transparent PNG (ComfyUI API format).

## Settings (optional environment variables)

| Variable | Purpose |
| --- | --- |
| `IMAGETOLAYERS_URL` | Use another server, for example a staging site. Default `https://www.imagetolayers.com`. |
| `IMAGETOLAYERS_SESSION_FILE` | Where the sign-in is stored. |

## Develop

```bash
python3 -m unittest discover -s tests   # client and layout; no ComfyUI or torch needed
```

## Limits

- Image up to 25 MB; the optional prompt is cut at 1000 characters.
- One image per run (the first image of a batch). A job can take a few minutes; the node waits up to 10 minutes.

## License

MIT, see [LICENSE](LICENSE). Using the nodes needs an Image to Layers account; the service itself is not covered by this license.

import os
import hashlib, uuid

try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property

from typing import Text, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from speechbrain.pretrained import EncoderClassifier

CACHE_DIR = os.getenv(
    "PYANNOTE_CACHE",
    os.path.expanduser("~/.cache/torch/pyannote"),
)

class SpeechBrainPretrainedSpeakerEmbedding:
    """Pretrained SpeechBrain speaker embedding

    Parameters
    ----------
    embedding : str
        Name of SpeechBrain model
    device : torch.device, optional
        Device
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`

    Usage
    -----
    >>> get_embedding = SpeechBrainPretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size

    >>> assert binary_masks.ndim == 1
    >>> assert binary_masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=binary_masks)
    """

    def __init__(
        self,
        embedding: Text = "speechbrain/spkrec-ecapa-voxceleb",
        device: torch.device = None,
        use_auth_token: Union[Text, None] = None,
        cach_dir: str = None,
    ):

        # if not SPEECHBRAIN_IS_AVAILABLE:
        #     raise ImportError(
        #         f"'speechbrain' must be installed to use '{embedding}' embeddings. "
        #         "Visit https://speechbrain.github.io for installation instructions."
        #     )

        super().__init__()
        self.embedding = embedding
        self.device = device
        self.cach_dir = cach_dir if cach_dir else CACHE_DIR
        
        self.classifier_ = EncoderClassifier.from_hparams(
            source=self.embedding,
            # savedir=f"{CACHE_DIR}/speechbrain",
            savedir=f"{self.cach_dir}/speechbrain-{hashlib.sha256(uuid.uuid4().hex.encode('utf-8')).hexdigest()}",
            run_opts={"device": self.device},
            use_auth_token=use_auth_token,
        )

    @cached_property
    def sample_rate(self) -> int:
        return self.classifier_.audio_normalizer.sample_rate

    @cached_property
    def dimension(self) -> int:
        dummy_waveforms = torch.rand(1, 16000).to(self.device)
        *_, dimension = self.classifier_.encode_batch(dummy_waveforms).shape
        return dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:

        lower, upper = 2, round(0.5 * self.sample_rate)
        middle = (lower + upper) // 2
        while lower + 1 < upper:
            try:
                _ = self.classifier_.encode_batch(
                    torch.randn(1, middle).to(self.device)
                )
                upper = middle
            except RuntimeError:
                lower = middle

            middle = (lower + upper) // 2

        return upper

    def __call__(
        self, waveforms: torch.Tensor, masks: torch.Tensor = None
    ) -> np.ndarray:
        """

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional

        Returns
        -------
        embeddings : (batch_size, dimension)

        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1

        waveforms = waveforms.squeeze(dim=1)

        if masks is None:
            signals = waveforms.squeeze(dim=1)
            wav_lens = signals.shape[1] * torch.ones(batch_size)

        else:

            batch_size_masks, _ = masks.shape
            assert batch_size == batch_size_masks

            # TODO: speed up the creation of "signals"
            # preliminary profiling experiments show
            # that it accounts for 15% of __call__
            # (the remaining 85% being the actual forward pass)

            imasks = F.interpolate(
                masks.unsqueeze(dim=1), size=num_samples, mode="nearest"
            ).squeeze(dim=1)

            imasks = imasks > 0.5

            signals = pad_sequence(
                [waveform[imask] for waveform, imask in zip(waveforms, imasks)],
                batch_first=True,
            )

            wav_lens = imasks.sum(dim=1)

        max_len = wav_lens.max()

        # corner case: every signal is too short
        if max_len < self.min_num_samples:
            return np.NAN * np.zeros((batch_size, self.dimension))

        too_short = wav_lens < self.min_num_samples
        wav_lens = wav_lens / max_len
        wav_lens[too_short] = 1.0

        embeddings = (
            self.classifier_.encode_batch(signals, wav_lens=wav_lens)
            .squeeze(dim=1)
            .cpu()
            .numpy()
        )

        embeddings[too_short.cpu().numpy()] = np.NAN

        return embeddings


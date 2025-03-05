# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file contains very basic regression tests for bitsandbytes
# It currently lives in the PEFT code base but should be moved to bnb eventually.
# These tests are very simplifistic and crude on purpose. If useful, they can be cleaned up and refactored later.

# Note that we make no assumptions about the correctness of the output, we only check that they didn't change
# unexpectedly.

# The expected values are generated by running the test until we have the `output`, then pass it to `bytes_from_tensor`

import io

import pytest
import torch
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          BitsAndBytesConfig)

bnb = pytest.importorskip("bitsandbytes")

device = torch.device("cuda")


def bytes_from_tensor(x):
    # helper function to create the expected output for regression testing
    f = io.BytesIO()
    torch.save(x, f)
    x_bytes = f.getvalue()
    f.close()
    return x_bytes


############
# OPT-125M #
############


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available.")
def test_opt_350m_4bit():
    torch.manual_seed(0)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float32,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-350m",
        quantization_config=bnb_config,
        torch_dtype=torch.float32,
    )

    input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(device)
    with torch.no_grad():
        output = model(input).logits[0, :3, :3].detach().cpu()

    expected_bytes = b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x12\x00archive/data.pklFB\x0e\x00ZZZZZZZZZZZZZZ\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\ttq\x05QK\x00K\x03K\x03\x86q\x06K\x03K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00'\x00archive/byteorderFB#\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00>\x00archive/data/0FB:\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\xfc\xd3\xff\xc00\xfe\xfe\xc0&eR@\x19j\x8d@,O\x1e?\xe9\xfb\x0bA\xcc\xb5OA\xc6?\xd6@\xd3\xc2\xe0@PK\x07\x08\xdb\xad]I$\x00\x00\x00$\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x1f\x00archive/versionFB\x1b\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0576858857385996278200001521679285581783PK\x07\x08\x93\x10\xf6E(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xdb\xad]I$\x00\x00\x00$\x00\x00\x00\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00archive/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf4\x01\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x93\x10\xf6E(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00R\x02\x00\x00archive/.data/serialization_idPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00B\x01\x00\x00\x00\x00\x00\x00\xf8\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00:\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x05\x00\x05\x00B\x01\x00\x00\xf8\x02\x00\x00\x00\x00"
    expected = torch.load(io.BytesIO(expected_bytes))
    torch.testing.assert_allclose(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available.")
def test_opt_350m_8bit():
    torch.manual_seed(0)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-350m",
        quantization_config=bnb_config,
        torch_dtype=torch.float32,
    )

    input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(device)
    with torch.no_grad():
        output = model(input).logits[0, :3, :3].detach().cpu()

    expected_bytes = b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x12\x00archive/data.pklFB\x0e\x00ZZZZZZZZZZZZZZ\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\ttq\x05QK\x00K\x03K\x03\x86q\x06K\x03K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00'\x00archive/byteorderFB#\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00>\x00archive/data/0FB:\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZN\t\xae\xbfR.\x8d\xbf\x88\xae\x01A@\x11\xb1@v\xae\x00@o\xc2\x14AJpNA-\x08\x0cACI\xf6@PK\x07\x08\xfe\xdb\xb9o$\x00\x00\x00$\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x1f\x00archive/versionFB\x1b\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0576858857385996278200001521667500867612PK\x07\x08\xb0\xb5\xcf\xfe(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xfe\xdb\xb9o$\x00\x00\x00$\x00\x00\x00\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00archive/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf4\x01\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xb0\xb5\xcf\xfe(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00R\x02\x00\x00archive/.data/serialization_idPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00B\x01\x00\x00\x00\x00\x00\x00\xf8\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00:\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x05\x00\x05\x00B\x01\x00\x00\xf8\x02\x00\x00\x00\x00"
    expected = torch.load(io.BytesIO(expected_bytes))
    torch.testing.assert_allclose(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available.")
def test_opt_350m_4bit_double_quant():
    torch.manual_seed(0)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-350m",
        quantization_config=bnb_config,
        torch_dtype=torch.float32,
    )

    input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(device)
    with torch.no_grad():
        output = model(input).logits[0, :3, :3].detach().cpu()

    expected_bytes = b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x12\x00archive/data.pklFB\x0e\x00ZZZZZZZZZZZZZZ\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\ttq\x05QK\x00K\x03K\x03\x86q\x06K\x03K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00'\x00archive/byteorderFB#\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00>\x00archive/data/0FB:\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ.\xe3\xfe\xc0H\xaa\xfe\xc0\xf6\x9aS@\xbe\x9c\x8b@\x06\x93\x1a?\xe8&\x0cA\x9f\x0cPA\xd4\xf4\xd6@V\xa3\xe1@PK\x07\x08J\x98\xbfQ$\x00\x00\x00$\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x1f\x00archive/versionFB\x1b\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0576858857385996278200001521700249059421PK\x07\x08\x9cW<\xe0(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00J\x98\xbfQ$\x00\x00\x00$\x00\x00\x00\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00archive/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf4\x01\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x9cW<\xe0(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00R\x02\x00\x00archive/.data/serialization_idPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00B\x01\x00\x00\x00\x00\x00\x00\xf8\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00:\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x05\x00\x05\x00B\x01\x00\x00\xf8\x02\x00\x00\x00\x00"
    expected = torch.load(io.BytesIO(expected_bytes))
    torch.testing.assert_allclose(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available.")
def test_opt_350m_4bit_compute_dtype_float16():
    torch.manual_seed(0)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-350m",
        quantization_config=bnb_config,
        torch_dtype=torch.float32,
    )

    input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(device)
    with torch.no_grad():
        output = model(input).logits[0, :3, :3].detach().cpu()

    expected_bytes = b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x12\x00archive/data.pklFB\x0e\x00ZZZZZZZZZZZZZZ\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\ttq\x05QK\x00K\x03K\x03\x86q\x06K\x03K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00'\x00archive/byteorderFB#\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00>\x00archive/data/0FB:\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\xfc\xd3\xff\xc00\xfe\xfe\xc0&eR@\x19j\x8d@,O\x1e?\xe9\xfb\x0bA\xcc\xb5OA\xc6?\xd6@\xd3\xc2\xe0@PK\x07\x08\xdb\xad]I$\x00\x00\x00$\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x1f\x00archive/versionFB\x1b\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0576858857385996278200001521679285581783PK\x07\x08\x93\x10\xf6E(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xdb\xad]I$\x00\x00\x00$\x00\x00\x00\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00archive/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf4\x01\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x93\x10\xf6E(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00R\x02\x00\x00archive/.data/serialization_idPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00B\x01\x00\x00\x00\x00\x00\x00\xf8\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00:\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x05\x00\x05\x00B\x01\x00\x00\xf8\x02\x00\x00\x00\x00"
    expected = torch.load(io.BytesIO(expected_bytes))
    torch.testing.assert_allclose(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available.")
def test_opt_350m_4bit_quant_type_nf4():
    torch.manual_seed(0)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-350m",
        quantization_config=bnb_config,
        torch_dtype=torch.float32,
    )

    input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(device)
    with torch.no_grad():
        output = model(input).logits[0, :3, :3].detach().cpu()

    expected_bytes = b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x12\x00archive/data.pklFB\x0e\x00ZZZZZZZZZZZZZZ\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\ttq\x05QK\x00K\x03K\x03\x86q\x06K\x03K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00'\x00archive/byteorderFB#\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00>\x00archive/data/0FB:\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ8\x18\xeb>\xd4\x82\x14\xbej\xbe\xff@:\xb9|@\x19\xb8\xb4?\xac\xae\x07A\x94iXA\xc8\x12\x13AHu\xdd@PK\x07\x08\xe1\xec\x0f\xf2$\x00\x00\x00$\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x1f\x00archive/versionFB\x1b\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0576858857385996278200001521529449342366PK\x07\x08\xbf\xb8\xd6H(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xe1\xec\x0f\xf2$\x00\x00\x00$\x00\x00\x00\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00archive/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf4\x01\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xbf\xb8\xd6H(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00R\x02\x00\x00archive/.data/serialization_idPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00B\x01\x00\x00\x00\x00\x00\x00\xf8\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00:\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x05\x00\x05\x00B\x01\x00\x00\xf8\x02\x00\x00\x00\x00"
    expected = torch.load(io.BytesIO(expected_bytes))
    torch.testing.assert_allclose(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available.")
def test_opt_350m_4bit_quant_storage():
    # note: using torch.float32 instead of the default torch.uint8 does not seem to affect the result
    torch.manual_seed(0)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_quant_storage=torch.float32,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-350m",
        quantization_config=bnb_config,
        torch_dtype=torch.float32,
    )

    input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(device)
    with torch.no_grad():
        output = model(input).logits[0, :3, :3].detach().cpu()

    expected_bytes = b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x12\x00archive/data.pklFB\x0e\x00ZZZZZZZZZZZZZZ\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\ttq\x05QK\x00K\x03K\x03\x86q\x06K\x03K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00'\x00archive/byteorderFB#\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00>\x00archive/data/0FB:\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\xfc\xd3\xff\xc00\xfe\xfe\xc0&eR@\x19j\x8d@,O\x1e?\xe9\xfb\x0bA\xcc\xb5OA\xc6?\xd6@\xd3\xc2\xe0@PK\x07\x08\xdb\xad]I$\x00\x00\x00$\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x1f\x00archive/versionFB\x1b\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0576858857385996278200001521679285581783PK\x07\x08\x93\x10\xf6E(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xdb\xad]I$\x00\x00\x00$\x00\x00\x00\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00archive/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf4\x01\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x93\x10\xf6E(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00R\x02\x00\x00archive/.data/serialization_idPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00B\x01\x00\x00\x00\x00\x00\x00\xf8\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00:\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x05\x00\x05\x00B\x01\x00\x00\xf8\x02\x00\x00\x00\x00"
    expected = torch.load(io.BytesIO(expected_bytes))
    torch.testing.assert_allclose(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available.")
def test_opt_350m_8bit_threshold():
    torch.manual_seed(0)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=3.0,  # default is 6.0
    )
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-350m",
        quantization_config=bnb_config,
        torch_dtype=torch.float32,
    )

    input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(device)
    with torch.no_grad():
        output = model(input).logits[0, :3, :3].detach().cpu()

    expected_bytes = b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x12\x00archive/data.pklFB\x0e\x00ZZZZZZZZZZZZZZ\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\ttq\x05QK\x00K\x03K\x03\x86q\x06K\x03K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00'\x00archive/byteorderFB#\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00>\x00archive/data/0FB:\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZR\xd5\x14\xc0\xc3\x9b\xf1\xbf \x9d\xde@D\x17\xc4@\t\xd1\x16@(\x97\x16A#TXA>\xdd\x12A\x08\x03\xfb@PK\x07\x08F\xd1\x87\xa3$\x00\x00\x00$\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x1f\x00archive/versionFB\x1b\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0576858857385996278200001521620583262466PK\x07\x08\x87\x89*\x93(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x99G\x1f\xb7\x9a\x00\x00\x00\x9a\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00F\xd1\x87\xa3$\x00\x00\x00$\x00\x00\x00\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00archive/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf4\x01\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x87\x89*\x93(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00R\x02\x00\x00archive/.data/serialization_idPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00B\x01\x00\x00\x00\x00\x00\x00\xf8\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00:\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x05\x00\x05\x00B\x01\x00\x00\xf8\x02\x00\x00\x00\x00"
    expected = torch.load(io.BytesIO(expected_bytes))
    torch.testing.assert_allclose(output, expected)


###########
# FLAN-T5 #
###########


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available.")
def test_flan_t5_4bit():
    torch.manual_seed(0)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float32,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base",
        quantization_config=bnb_config,
        torch_dtype=torch.float32,
    )

    input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input, return_dict_in_generate=True, output_scores=True
        )
        output = output.scores[0][0, :10].detach().cpu()

    expected_bytes = b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x12\x00archive/data.pklFB\x0e\x00ZZZZZZZZZZZZZZ\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\ntq\x05QK\x00K\n\x85q\x06K\x01\x85q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\x19\xea\x16n\x96\x00\x00\x00\x96\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00+\x00archive/byteorderFB'\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00>\x00archive/data/0FB:\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZpb\x0f\xc2\x91\xa3\x85\xc0\x86\xee\x83\xc0\xae\xea\xdc?F\xad-\xc1\xe4*k\xc0\x12\x84\x86\xc09\xf9\xc8\xc0|\x861\xc0m\xf7\x0c\xc1PK\x07\x08\xf1y:\xda(\x00\x00\x00(\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x1b\x00archive/versionFB\x17\x00ZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0576858857385996278200001223527302082336PK\x07\x08~n}q(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x19\xea\x16n\x96\x00\x00\x00\x96\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe6\x00\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xf1y:\xda(\x00\x00\x00(\x00\x00\x00\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00archive/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf8\x01\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00~n}q(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00R\x02\x00\x00archive/.data/serialization_idPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00B\x01\x00\x00\x00\x00\x00\x00\xf8\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00:\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x05\x00\x05\x00B\x01\x00\x00\xf8\x02\x00\x00\x00\x00"
    expected = torch.load(io.BytesIO(expected_bytes))
    torch.testing.assert_allclose(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available.")
@pytest.mark.xfail  # might not be reproducible depending on hardware
def test_flan_t5_8bit():
    torch.manual_seed(0)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base",
        quantization_config=bnb_config,
        torch_dtype=torch.float32,
    )

    input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input, return_dict_in_generate=True, output_scores=True
        )
        output = output.scores[0][0, :10].detach().cpu()

    expected_bytes = b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x12\x00archive/data.pklFB\x0e\x00ZZZZZZZZZZZZZZ\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\ntq\x05QK\x00K\n\x85q\x06K\x01\x85q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\x19\xea\x16n\x96\x00\x00\x00\x96\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00+\x00archive/byteorderFB'\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00>\x00archive/data/0FB:\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\xebd)\xc2\xac\x1c\xba\xc0F\x0c\xbf\xc0v\\\x88?\x9f\x7fW\xc1H\xbd\xa0\xc0\xf4\xaf\xaf\xc0@:\x02\xc1\xbcjr\xc0\xf7\x95$\xc1PK\x07\x08\x12\xcc\x86\x12(\x00\x00\x00(\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x1b\x00archive/versionFB\x17\x00ZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0576858857385996278200001226216142756281PK\x07\x08\xa0Z\xf3\xd2(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x19\xea\x16n\x96\x00\x00\x00\x96\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe6\x00\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x12\xcc\x86\x12(\x00\x00\x00(\x00\x00\x00\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00archive/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf8\x01\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xa0Z\xf3\xd2(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00R\x02\x00\x00archive/.data/serialization_idPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00B\x01\x00\x00\x00\x00\x00\x00\xf8\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00:\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x05\x00\x05\x00B\x01\x00\x00\xf8\x02\x00\x00\x00\x00"
    expected = torch.load(io.BytesIO(expected_bytes))
    torch.testing.assert_allclose(output, expected)

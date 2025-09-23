# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import os
import numpy as np
import rebel
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def _get_index(self, name):
        parts = name.split("__")
        return int(parts[1])

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        # Configure input_dict
        self.input_dict = {}
        for config_input in model_config["input"]:
            index = self._get_index(config_input["name"])
            self.input_dict[index] = [
                config_input["name"],
                config_input["data_type"],
                config_input["dims"],
            ]
        # Configure output_dict
        self.output_dict = {}
        for config_output in model_config["output"]:
            index = self._get_index(config_output["name"])
            self.output_dict[index] = [
                config_output["name"],
                config_output["data_type"],
                config_output["dims"],
            ]
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT__0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        rbln_path = os.path.join(
            args["model_repository"],
            args["model_version"],
            f"{args['model_name']}.rbln",
        )
        self.model_name = args["model_name"]
        # Load compiled model
        compiled_model = rebel.RBLNCompiledModel(
            rbln_path
        )
        # Create runner for each batch size
        self.runner = rebel.Runtime(compiled_model)

    def execute(self, requests):
        # Preprocess the input data
        responses = []
        inputs = []
        num_requests = len(requests)
        request_batch_sizes = []
        for i in self.input_dict.keys():
            name, dt, _ = self.input_dict[i]
            first_tensor = pb_utils.get_input_tensor_by_name(
                requests[0], name
            ).as_numpy()
            request_batch_sizes.append(first_tensor.shape[0])
            batched_tensor = first_tensor
            for j in range(1, num_requests):
                tensor = pb_utils.get_input_tensor_by_name(requests[j], name).as_numpy()
                request_batch_sizes.append(request_batch_sizes[-1] + tensor.shape[0])
                batched_tensor = np.concatenate((batched_tensor, tensor), axis=0)
            inputs.append(batched_tensor)
        batch_size = batched_tensor.shape[0]
        # Run inference on the RBLN model
        batched_results = self.runner(batched_tensor)
        # Postprocess the output data
        chunky_batched_results = []
        for i in self.output_dict.keys():
            batch = (
                batched_results[i]
                if (isinstance(batched_results, tuple) or
                    isinstance(batched_results, list))
                else batched_results
            )
            chunky_batched_results.append(
                np.array_split(batch, request_batch_sizes, axis=0)
            )
        # Send response
        for i in range(num_requests):
            output_tensors = []
            for j in self.output_dict.keys():
                name, dt, _ = self.output_dict[j]
                result = chunky_batched_results[j][i]
                output_tensor = pb_utils.Tensor(
                    name, result.astype(pb_utils.triton_string_to_numpy(dt))
                )
                output_tensors.append(output_tensor)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up...")

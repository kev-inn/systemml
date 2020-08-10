/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.instructions.fed;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;

import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.transform.decode.Decoder;
import org.apache.sysds.runtime.transform.decode.DecoderFactory;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class ParameterizedBuiltinFEDInstruction extends ComputationFEDInstruction {
	protected final LinkedHashMap<String, String> params;

	protected ParameterizedBuiltinFEDInstruction(Operator op, LinkedHashMap<String, String> paramsMap, CPOperand out,
		String opcode, String istr) {
		super(FEDInstruction.FEDType.ParameterizedBuiltin, op, null, null, out, opcode, istr);
		params = paramsMap;
	}

	public HashMap<String, String> getParameterMap() {
		return params;
	}

	public String getParam(String key) {
		return getParameterMap().get(key);
	}

	public static LinkedHashMap<String, String> constructParameterMap(String[] params) {
		// process all elements in "params" except first(opcode) and last(output)
		LinkedHashMap<String, String> paramMap = new LinkedHashMap<>();

		// all parameters are of form <name=value>
		String[] parts;
		for(int i = 1; i <= params.length - 2; i++) {
			parts = params[i].split(Lop.NAME_VALUE_SEPARATOR);
			paramMap.put(parts[0], parts[1]);
		}

		return paramMap;
	}

	public static ParameterizedBuiltinFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// first part is always the opcode
		String opcode = parts[0];
		// last part is always the output
		CPOperand out = new CPOperand(parts[parts.length - 1]);

		// process remaining parts and build a hash map
		LinkedHashMap<String, String> paramsMap = constructParameterMap(parts);

		// determine the appropriate value function
		ValueFunction func = null;
		if(opcode.equals("transformapply") || opcode.equals("transformdecode")) {
			return new ParameterizedBuiltinFEDInstruction(null, paramsMap, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode (" + opcode + ") for ParameterizedBuiltin Instruction.");
		}
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		if(opcode.equalsIgnoreCase("transformdecode")) {
			// acquire locks
			MatrixObject data = ec.getMatrixObject(params.get("target"));
			FrameBlock meta = ec.getFrameInput(params.get("meta"));
			String spec = params.get("spec");

			Decoder decoder = DecoderFactory
				.createDecoder(spec, meta.getColumnNames(), null, meta, (int) data.getNumColumns());

			Map<FederatedRange, FederatedData> fedMapping = data.getFedMapping();

			ExecutorService pool = CommonThreadPool.get(fedMapping.size());
			Map<FederatedRange, FederatedData> decodedMapping = new HashMap<>();
			ArrayList<FederatedDecodeTask> createTasks = new ArrayList<>();
			for(Map.Entry<FederatedRange, FederatedData> fedMap : fedMapping.entrySet())
				createTasks
					.add(new FederatedDecodeTask(fedMap.getKey(), fedMap.getValue(), decoder, meta, decodedMapping));
			CommonThreadPool.invokeAndShutdown(pool, createTasks);

			// construct a federated matrix with the encoded data
			FrameObject decodedFrame = ec.getFrameObject(output);
			decodedFrame.setSchema(decoder.getSchema());
			decodedFrame.getDataCharacteristics().set(data.getDataCharacteristics());
			decodedFrame.getDataCharacteristics().setCols(decoder.getSchema().length);
			// set the federated mapping for the matrix
			decodedFrame.setFedMapping(decodedMapping);

			// release locks
			ec.releaseFrameInput(params.get("meta"));
		}
		else {
			throw new DMLRuntimeException("Unknown opcode : " + opcode);
		}
	}

	private static class FederatedDecodeTask implements Callable<Void> {
		private final FederatedRange _range;
		private final FederatedData _data;
		private final Decoder _globalDecoder;
		private final FrameBlock _meta;
		private final Map<FederatedRange, FederatedData> _resultMapping;

		public FederatedDecodeTask(FederatedRange range, FederatedData data, Decoder globalDecoder, FrameBlock meta,
			Map<FederatedRange, FederatedData> resultMapping) {
			_range = range;
			_data = data;
			_globalDecoder = globalDecoder;
			_meta = meta;
			_resultMapping = resultMapping;
		}

		@Override
		public Void call() throws Exception {
			// compute the range in the decoded FrameBlock, this encoded range aligns to
			long[] beginDims = Arrays.copyOf(_range.getBeginDims(), _range.getBeginDims().length);
			long[] endDims = Arrays.copyOf(_range.getEndDims(), _range.getEndDims().length);
			int colStartBefore = (int) beginDims[1];

			// update begin end dims (column part) considering columns added by dummycoding
			_globalDecoder.updateIndexRanges(beginDims, endDims);
			/*
			 * int lowerColDest = (int) _range.getBeginDims()[1] + 1; int upperColDest = (int) _range.getEndDims()[1] +
			 * 1; int dummycodedOffset = 0; for(int i = 0, dcColCounter = 1; dcColCounter < upperColDest; i++,
			 * dcColCounter++) { long numDistinct = _meta.getColumnMetadata(i).getNumDistinct();
			 * 
			 * if(dcColCounter < lowerColDest) { if(numDistinct > 0) { dummycodedOffset += numDistinct; lowerColDest -=
			 * numDistinct - 1; } else dummycodedOffset++; }
			 * 
			 * if(numDistinct > 0) upperColDest -= numDistinct - 1; dcColCounter += numDistinct; }
			 */
			FrameBlock meta = new FrameBlock();
			synchronized(_meta) {
				_meta.slice(0, _meta.getNumRows() - 1, (int) beginDims[1], (int) endDims[1] - 1, meta);
			}

			// get the decoder segment that is relevant for this federated worker
			Decoder decoder = _globalDecoder
				.subRangeDecoder((int) beginDims[1] + 1, (int) endDims[1] + 1, colStartBefore);

			FederatedResponse response = _data
				.executeFederatedOperation(new FederatedRequest(FederatedRequest.FedMethod.DECODE, decoder, meta), true)
				.get();

			long varId = (long) response.getData()[0];
			synchronized(_resultMapping) {
				_resultMapping.put(new FederatedRange(beginDims, endDims), new FederatedData(_data, varId));
			}
			return null;
		}
	}
}

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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;

import org.apache.sysds.common.Types;
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

			Map<FederatedRange, FederatedData> fedMapping = data.getFedMapping();

			ExecutorService pool = CommonThreadPool.get(fedMapping.size());
			Types.ValueType[] schema = new Types.ValueType[(int) data.getNumColumns()];
			Map<FederatedRange, FederatedData> decodedMapping = new HashMap<>();
			ArrayList<FederatedDecodeTask> createTasks = new ArrayList<>();
			for(Map.Entry<FederatedRange, FederatedData> fedMap : fedMapping.entrySet())
				createTasks
					.add(new FederatedDecodeTask(fedMap.getKey(), fedMap.getValue(), meta, spec, decodedMapping, schema));
			CommonThreadPool.invokeAndShutdown(pool, createTasks);

			// construct a federated matrix with the encoded data
			FrameObject decodedFrame = ec.getFrameObject(output);
			decodedFrame.setSchema(schema);
			decodedFrame.getDataCharacteristics().set(data.getDataCharacteristics());
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
		private final FrameBlock _meta;
		private final String _spec;
		private final Map<FederatedRange, FederatedData> _resultMapping;
		private final Types.ValueType[] _schema;

		public FederatedDecodeTask(FederatedRange range, FederatedData data, FrameBlock meta, String spec,
			Map<FederatedRange, FederatedData> resultMapping, Types.ValueType[] schema) {
			_range = range;
			_data = data;
			_meta = meta;
			_spec = spec;
			_resultMapping = resultMapping;
			_schema = schema;
		}

		@Override
		public Void call() throws Exception {
			int columnOffset = (int) _range.getBeginDims()[1] + 1;

			FrameBlock meta = new FrameBlock();
			synchronized(_meta) {
				_meta.slice(0, _meta.getNumRows() - 1, columnOffset - 1, (int) _range.getEndDims()[1] - 1, meta);
			}

			FederatedResponse response = _data.executeFederatedOperation(
				new FederatedRequest(FederatedRequest.FedMethod.DECODE, meta, _spec, columnOffset),
				true).get();

			long varId = (long) response.getData()[0];
			synchronized(_resultMapping) {
				_resultMapping.put(new FederatedRange(_range), new FederatedData(_data, varId));
			}
			Types.ValueType[] subSchema = (Types.ValueType[]) response.getData()[1];
			synchronized (_schema) {
				// It would be possible to assert that different federated workers don't give different value types
				// for the same columns, but the performance impact is not worth the effort
				System.arraycopy(subSchema, 0, _schema, columnOffset - 1, subSchema.length);
			}
			return null;
		}
	}
}

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

package org.apache.sysds.test.functions.transform;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class TransformFederatedEncodeDecodeTest extends AutomatedTestBase {
	private static final String TEST_NAME1 = "TransformFederatedEncodeDecode";
	private static final String TEST_DIR = "functions/transform/";
	private static final String TEST_CLASS_DIR = TEST_DIR + TransformFederatedEncodeDecodeTest.class.getSimpleName()
		+ "/";

	private static final String SPEC = "TransformEncodeDecodeSpec.json";

	private static final int rows = 1234;
	private static final int cols = 2;
	private static final double sparsity1 = 0.9;
	private static final double sparsity2 = 0.1;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"FO"}));
	}

	@Test
	public void runTestCSVDenseCP() {
		runTransformEncodeDecodeTest(false, Types.FileFormat.CSV);
	}

	@Test
	public void runTestCSVSparseCP() {
		runTransformEncodeDecodeTest(true, Types.FileFormat.CSV);
	}

	@Test
	public void runTestTextcellDenseCP() {
		runTransformEncodeDecodeTest(false, Types.FileFormat.TEXT);
	}

	@Test
	public void runTestTextcellSparseCP() {
		runTransformEncodeDecodeTest(true, Types.FileFormat.TEXT);
	}

	@Test
	public void runTestBinaryDenseCP() {
		runTransformEncodeDecodeTest(false, Types.FileFormat.BINARY);
	}

	@Test
	public void runTestBinarySparseCP() {
		runTransformEncodeDecodeTest(true, Types.FileFormat.BINARY);
	}

	private void runTransformEncodeDecodeTest(boolean sparse, Types.FileFormat format) {
		ExecMode platformOld = rtplatform;
		rtplatform = ExecMode.SINGLE_NODE;

		Thread t1 = null, t2 = null;
		try {
			getAndLoadTestConfiguration(TEST_NAME1);

			int port1 = getRandomAvailablePort();
			t1 = startLocalFedWorker(port1);
			int port2 = getRandomAvailablePort();
			t2 = startLocalFedWorker(port2);

			// schema
			Types.ValueType[] schema = new Types.ValueType[cols / 2];
			Arrays.fill(schema, Types.ValueType.FP64);
			// generate and write input data
			// A is the data that will be aggregated and not recoded
			double[][] A = TestUtils.round(getRandomMatrix(rows, cols / 2, 1, 15, sparse ? sparsity2 : sparsity1, 7));
			writeInputFrameWithMTD("A", A, false, schema, format);

			// B will be recoded and will be the column that will be grouped by
			Arrays.fill(schema, Types.ValueType.STRING);
			// we set sparsity to 1.0 to ensure all the string labels exist
			double[][] B = TestUtils.round(getRandomMatrix(rows, cols / 2, 1, 15, 1.0, 8));
			writeInputFrameWithMTD("B", B, false, schema, format);

			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME1 + ".dml";

			programArgs = new String[] {"-explain", "-args", TestUtils.federatedAddress("localhost", port1, input("A")),
				TestUtils.federatedAddress("localhost", port1, input("B")), Integer.toString(rows),
				Integer.toString(cols / 2), Integer.toString(cols), SCRIPT_DIR + TEST_DIR + SPEC, output("FO"),
				format.toString()};

			// run test
			runTest(true, false, null, -1);

			// compare matrices (values recoded to identical codes)
			FrameReader reader = FrameReaderFactory.createFrameReader(format);
			FrameBlock FO = reader.readFrameFromHDFS(output("FO"), 15, 2);
			HashMap<String, Long> cFA = getCounts(A, B);
			Iterator<String[]> iterFO = FO.getStringRowIterator();
			while(iterFO.hasNext()) {
				String[] row = iterFO.next();
				Double expected = (double) cFA.get(row[1]);
				Double val = (row[0] != null) ? Double.parseDouble(row[0]) : 0;
				Assert.assertEquals("Output aggregates don't match: " + expected + " vs " + val, expected, val);
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			Assert.fail(ex.getMessage());
		}
		finally {
			TestUtils.shutdownThread(t1);
			TestUtils.shutdownThread(t2);
			rtplatform = platformOld;
		}
	}

	private static HashMap<String, Long> getCounts(double[][] countFrame, double[][] groupFrame) {
		HashMap<String, Long> ret = new HashMap<>();
		for(int i = 0; i < countFrame.length; i++) {
			String key = "Str" + groupFrame[i][0];
			Long tmp = ret.get(key);
			ret.put(key, (tmp != null) ? tmp + 1 : 1);
		}
		return ret;
	}
}

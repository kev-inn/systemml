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

package org.apache.sysds.runtime.transform.encode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class EncoderDummycode extends Encoder 
{
	private static final long serialVersionUID = 5832130477659116489L;

	public int[] _domainSizes = null;  // length = #of dummycoded columns
	private long _dummycodedLength = 0; // #of columns after dummycoded

	public EncoderDummycode(JSONObject parsedSpec, String[] colnames, int clen, int minCol, int maxCol)
		throws JSONException {
		super(null, clen);

		if(parsedSpec.containsKey(TfMethod.DUMMYCODE.toString())) {
			int[] collist = TfMetaUtils
				.parseJsonIDList(parsedSpec, colnames, TfMethod.DUMMYCODE.toString(), minCol, maxCol);
			initColList(collist);
		}
	}

	public EncoderDummycode() {
		super(null, 0);
	}
	
	@Override
	public int getNumCols() {
		return (int)_dummycodedLength;
	}
	
	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		return apply(in, out);
	}

	@Override
	public void build(FrameBlock in) {
		//do nothing
	}
	
	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		//allocate output in dense or sparse representation
		final boolean sparse = MatrixBlock.evalSparseFormatInMemory(
			out.getNumRows(), getNumCols(), out.getNonZeros());
		MatrixBlock ret = new MatrixBlock(out.getNumRows(), getNumCols(), sparse);
		
		//append dummy coded or unchanged values to output
		final int clen = out.getNumColumns();
		for( int i=0; i<out.getNumRows(); i++ ) {
			for(int colID=1, idx=0, ncolID=1; colID <= clen; colID++) {
				double val = out.quickGetValue(i, colID-1);
				if( idx < _colList.length && colID==_colList[idx] ) {
					ret.appendValue(i, ncolID-1+(int)val-1, 1);
					ncolID += _domainSizes[idx];
					idx ++;
				}
				else {
					double ptval = out.quickGetValue(i, colID-1);
					ret.appendValue(i, ncolID-1, ptval);
					ncolID ++;
				}
			}
		}
		return ret;
	}

	@Override
	public Encoder subRangeEncoder(int colStart, int colEnd) {
		if(colStart - 1 >= _clen)
			return null;

		List<Integer> cols = new ArrayList<>();
		List<Integer> domainSizes = new ArrayList<>();
		int newDummycodedLength = colEnd - colStart;
		for(int i = 0; i < _colList.length; i++){
			int col = _colList[i];
			if(col >= colStart && col < colEnd) {
				// add the correct column, removed columns before start
				// colStart - 1 because colStart is 1-based
				int corrColumn = col - (colStart - 1);
				cols.add(corrColumn);
				domainSizes.add(_domainSizes[i]);
				newDummycodedLength += _domainSizes[i] - 1;
			}
		}
		if(cols.isEmpty())
			// empty encoder -> sub range encoder does not exist
			return null;

		EncoderDummycode subRangeEncoder = new EncoderDummycode();
		subRangeEncoder._clen = colEnd - colStart;
		subRangeEncoder._colList = cols.stream().mapToInt(i -> i).toArray();
		subRangeEncoder._domainSizes = domainSizes.stream().mapToInt(i -> i).toArray();
		subRangeEncoder._dummycodedLength = newDummycodedLength;
		return subRangeEncoder;
	}

	@Override
	public void mergeAt(Encoder other, int col) {
		if(other instanceof EncoderDummycode) {
			// assure col lists exist
			if(_colList == null) {
				_clen = 0;
				_colList = new int[0];
			}
			if(other._colList == null) {
				other._clen = 0;
				other._colList = new int[0];
			}
			mergeColumnInfo(other, col);

			_domainSizes = new int[_colList.length];
			_dummycodedLength = _clen;
			for( int j=0; j<_colList.length; j++ ) {
				// temporary, will be updated later
				_domainSizes[j] = 1;
			}
			return;
		}
		super.mergeAt(other, col);
	}
	
	@Override
	public void updateIndexRanges(long[] beginDims, long[] endDims) {
		if(_colList == null)
			return;

		for(int i = 0; i < _colList.length; i++) {
			// 1-based vs 0-based
			if(_colList[i] < beginDims[1] + 1) {
				// new columns inserted left of the columns of this partial (federated) block
				beginDims[1] += _domainSizes[i] - 1;
				endDims[1] += _domainSizes[i] - 1;
			}
			else if(_colList[i] < endDims[1] + 1) {
				// new columns inserted in this (federated) block
				endDims[1] += _domainSizes[i] - 1;
			}
		}
	}
	
	public void updateDomainSizes(List<Encoder> encoders) {
		// maps the column ids of the columns encoded by this Dummycode Encoder to their respective indexes
		// in the _colList
		Map<Integer, Integer> colIDToIxMap = new HashMap<>();
		for (int i = 0; i < _colList.length; i++)
			colIDToIxMap.put(_colList[i], i);
		
		for (Encoder encoder : encoders) {
			int[] distinct = null;
			if (encoder instanceof EncoderRecode) {
				EncoderRecode encoderRecode = (EncoderRecode) encoder;
				distinct = encoderRecode.numDistinctValues();
			}
			else if (encoder instanceof EncoderBin) {
				distinct = ((EncoderBin) encoder)._numBins;
			}
			
			if (distinct != null) {
				// search for match of encoded columns
				for (int i = 0; i < encoder._colList.length; i++) {
					Integer ix = colIDToIxMap.get(encoder._colList[i]);
					
					if (ix != null) {
						// set size
						_domainSizes[ix] = distinct[i];
						_dummycodedLength += _domainSizes[ix] - 1;
					}
				}
			}
		}
	}

	@Override
	public FrameBlock getMetaData(FrameBlock out) {
		return out;
	}
	
	@Override
	public void initMetaData(FrameBlock meta) {
		//initialize domain sizes and output num columns
		_domainSizes = new int[_colList.length];
		_dummycodedLength = _clen;
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j]; //1-based
			_domainSizes[j] = (int)meta.getColumnMetadata()[colID-1].getNumDistinct();
			_dummycodedLength += _domainSizes[j]-1;
		}
	}
	
	@Override
	public MatrixBlock getColMapping(FrameBlock meta, MatrixBlock out) {
		final int clen = out.getNumRows();
		for(int colID=1, idx=0, ncolID=1; colID <= clen; colID++) {
			int start = ncolID;
			if( idx < _colList.length && colID==_colList[idx] ) {
				ncolID += meta.getColumnMetadata(colID-1).getNumDistinct();
				idx ++;
			}
			else {
				ncolID ++;
			}
			out.quickSetValue(colID-1, 0, colID);
			out.quickSetValue(colID-1, 1, start);
			out.quickSetValue(colID-1, 2, ncolID-1);
		}
		
		return out;
	}
}

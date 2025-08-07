/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.lancedb.lance.spark.utils;

import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.util.ArrayData;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.HashMap;
import java.util.Map;

/** Validates vector columns have the correct dimensions as specified in metadata. */
public class VectorValidator {

  private static final String ARROW_FIXED_SIZE_LIST_SIZE_KEY = "arrow.FixedSizeList.size";

  private final Map<Integer, VectorFieldInfo> vectorFields;

  public VectorValidator(StructType schema) {
    this.vectorFields = extractVectorFields(schema);
  }

  /**
   * Validates that all vector columns in the row have the correct dimensions.
   *
   * @param row The row to validate
   * @throws IllegalArgumentException if any vector has incorrect dimensions
   */
  public void validateRow(InternalRow row) {
    for (VectorFieldInfo fieldInfo : vectorFields.values()) {
      if (!row.isNullAt(fieldInfo.ordinal)) {
        ArrayData arrayData = row.getArray(fieldInfo.ordinal);
        long actualDimension = arrayData.numElements();

        if (actualDimension != fieldInfo.expectedDimension) {
          throw new IllegalArgumentException(
              String.format(
                  "Vector column '%s' expected dimension %d but got %d at row position %d",
                  fieldInfo.name, fieldInfo.expectedDimension, actualDimension, fieldInfo.ordinal));
        }
      }
    }
  }

  /** Checks if there are any vector fields to validate. */
  public boolean hasVectorFields() {
    return !vectorFields.isEmpty();
  }

  private Map<Integer, VectorFieldInfo> extractVectorFields(StructType schema) {
    Map<Integer, VectorFieldInfo> fields = new HashMap<>();
    StructField[] structFields = schema.fields();

    for (int i = 0; i < structFields.length; i++) {
      StructField field = structFields[i];
      if (isVectorField(field)) {
        long dimension = getVectorDimension(field.metadata());
        fields.put(i, new VectorFieldInfo(field.name(), i, dimension));
      }
    }

    return fields;
  }

  private boolean isVectorField(StructField field) {
    DataType dataType = field.dataType();
    Metadata metadata = field.metadata();

    // Check if it's an array type with float or double elements
    if (!(dataType instanceof ArrayType)) {
      return false;
    }

    ArrayType arrayType = (ArrayType) dataType;
    DataType elementType = arrayType.elementType();

    if (!(elementType instanceof FloatType || elementType instanceof DoubleType)) {
      return false;
    }

    // Check metadata for fixed size list specification
    return metadata.contains(ARROW_FIXED_SIZE_LIST_SIZE_KEY)
        && metadata.getLong(ARROW_FIXED_SIZE_LIST_SIZE_KEY) > 0;
  }

  private long getVectorDimension(Metadata metadata) {
    return metadata.getLong(ARROW_FIXED_SIZE_LIST_SIZE_KEY);
  }

  private static class VectorFieldInfo {
    final String name;
    final int ordinal;
    final long expectedDimension;

    VectorFieldInfo(String name, int ordinal, long expectedDimension) {
      this.name = name;
      this.ordinal = ordinal;
      this.expectedDimension = expectedDimension;
    }
  }
}

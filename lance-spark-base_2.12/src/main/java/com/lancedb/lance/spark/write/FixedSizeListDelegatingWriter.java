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
package com.lancedb.lance.spark.write;

import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.execution.arrow.ArrowWriter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Wraps Spark's ArrowWriter to handle FixedSizeList conversion. Creates a temporary schema with
 * regular Lists, lets Spark write to it, then converts to FixedSizeList.
 */
public class FixedSizeListDelegatingWriter extends ArrowWriter {

  private final VectorSchemaRoot targetRoot;
  private final VectorSchemaRoot tempRoot;
  private final ArrowWriter sparkWriter;
  private final Map<String, Integer> fixedSizeListFields;

  public FixedSizeListDelegatingWriter(VectorSchemaRoot targetRoot) {
    super(targetRoot, new org.apache.spark.sql.execution.arrow.ArrowFieldWriter[0]);
    this.targetRoot = targetRoot;
    this.fixedSizeListFields = new HashMap<>();

    System.out.println("FixedSizeListDelegatingWriter: Target schema = " + targetRoot.getSchema());

    // Create temp schema with regular Lists instead of FixedSizeLists
    List<Field> tempFields = new ArrayList<>();
    for (Field field : targetRoot.getSchema().getFields()) {
      if (field.getType() instanceof ArrowType.FixedSizeList) {
        ArrowType.FixedSizeList fsl = (ArrowType.FixedSizeList) field.getType();
        fixedSizeListFields.put(field.getName(), fsl.getListSize());
        System.out.println(
            "Found FixedSizeList field: " + field.getName() + " with size " + fsl.getListSize());

        // Replace with regular List
        Field listField =
            new Field(
                field.getName(),
                new FieldType(field.isNullable(), ArrowType.List.INSTANCE, field.getDictionary()),
                field.getChildren());
        tempFields.add(listField);
      } else {
        tempFields.add(field);
      }
    }

    Schema tempSchema = new Schema(tempFields);
    this.tempRoot = VectorSchemaRoot.create(tempSchema, targetRoot.getVector(0).getAllocator());
    this.sparkWriter = ArrowWriter.create(tempRoot);
  }

  @Override
  public void write(InternalRow row) {
    // Let Spark write to temp root with regular Lists
    sparkWriter.write(row);
  }

  @Override
  public void finish() {
    sparkWriter.finish();

    // Copy data from temp root to target root, converting Lists to FixedSizeLists
    int rowCount = tempRoot.getRowCount();
    System.out.println(
        "FixedSizeListDelegatingWriter.finish: rowCount = "
            + rowCount
            + ", fixedSizeListFields = "
            + fixedSizeListFields);
    targetRoot.allocateNew();

    for (Field field : targetRoot.getSchema().getFields()) {
      if (fixedSizeListFields.containsKey(field.getName())) {
        // Convert List to FixedSizeList
        ListVector source = (ListVector) tempRoot.getVector(field.getName());
        FixedSizeListVector target = (FixedSizeListVector) targetRoot.getVector(field.getName());
        convertListToFixedSizeList(source, target, rowCount);
      } else {
        // Direct copy for other fields
        targetRoot.getVector(field.getName()).copyFrom(0, 0, tempRoot.getVector(field.getName()));
        targetRoot.getVector(field.getName()).setValueCount(rowCount);
      }
    }

    targetRoot.setRowCount(rowCount);
  }

  private void convertListToFixedSizeList(
      ListVector source, FixedSizeListVector target, int rowCount) {
    for (int i = 0; i < rowCount; i++) {
      if (source.isNull(i)) {
        target.setNull(i);
      } else {
        int start = source.getElementStartIndex(i);
        int end = source.getElementEndIndex(i);

        if (source.getDataVector() instanceof Float4Vector) {
          Float4Vector sourceData = (Float4Vector) source.getDataVector();
          Float4Vector targetData = (Float4Vector) target.getDataVector();

          int targetIndex = i * target.getListSize();
          for (int j = start; j < end; j++) {
            targetData.setSafe(targetIndex++, sourceData.get(j));
          }
        } else if (source.getDataVector() instanceof Float8Vector) {
          Float8Vector sourceData = (Float8Vector) source.getDataVector();
          Float8Vector targetData = (Float8Vector) target.getDataVector();

          int targetIndex = i * target.getListSize();
          for (int j = start; j < end; j++) {
            targetData.setSafe(targetIndex++, sourceData.get(j));
          }
        }
      }
    }
    target.setValueCount(rowCount);
  }

  @Override
  public void reset() {
    sparkWriter.reset();
    tempRoot.clear();
    targetRoot.clear();
  }
}

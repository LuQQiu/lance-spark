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
package org.apache.spark.sql.lance

import org.apache.arrow.vector.types.pojo.ArrowType
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.LanceArrowUtils
import org.scalatest.funsuite.AnyFunSuite

class VectorColumnTest extends AnyFunSuite {

  test("ArrayType with fixed size metadata should convert to FixedSizeList") {
    // Create a schema with fixed size list metadata
    val metadata = new MetadataBuilder()
      .putLong("arrow.FixedSizeList.size", 128)
      .build()

    val schema = StructType(Seq(
      StructField("id", LongType, nullable = false),
      StructField(
        "embeddings",
        ArrayType(FloatType, containsNull = false),
        nullable = true,
        metadata)))

    // Convert to Arrow schema
    val arrowSchema =
      LanceArrowUtils.toArrowSchema(schema, "UTC", errorOnDuplicatedFieldNames = true)

    // Check the embeddings field
    val embeddingsField = arrowSchema.findField("embeddings")
    assert(embeddingsField != null)

    // Verify it's a FixedSizeList with dimension 128
    embeddingsField.getType match {
      case fsl: ArrowType.FixedSizeList =>
        assert(fsl.getListSize == 128, s"Expected dimension 128, got ${fsl.getListSize}")
      case other =>
        fail(s"Expected FixedSizeList, got ${other.getClass.getSimpleName}")
    }
  }

  test("ArrayType without fixed size metadata should remain as List") {
    val schema = StructType(Seq(
      StructField("id", LongType, nullable = false),
      StructField("values", ArrayType(FloatType, containsNull = false), nullable = true)))

    // Convert to Arrow schema
    val arrowSchema =
      LanceArrowUtils.toArrowSchema(schema, "UTC", errorOnDuplicatedFieldNames = true)

    // Check the values field
    val valuesField = arrowSchema.findField("values")
    assert(valuesField != null)

    // Verify it's a regular List
    valuesField.getType match {
      case _: ArrowType.List =>
      // Success - it's a regular list
      case other =>
        fail(s"Expected List, got ${other.getClass.getSimpleName}")
    }
  }

  test("DoubleType fixed size lists should also work") {
    val metadata = new MetadataBuilder()
      .putLong("arrow.FixedSizeList.size", 256)
      .build()

    val schema = StructType(Seq(
      StructField(
        "features",
        ArrayType(DoubleType, containsNull = false),
        nullable = true,
        metadata)))

    val arrowSchema =
      LanceArrowUtils.toArrowSchema(schema, "UTC", errorOnDuplicatedFieldNames = true)
    val featuresField = arrowSchema.findField("features")

    featuresField.getType match {
      case fsl: ArrowType.FixedSizeList =>
        assert(fsl.getListSize == 256)
      case other =>
        fail(s"Expected FixedSizeList, got ${other.getClass.getSimpleName}")
    }
  }
}

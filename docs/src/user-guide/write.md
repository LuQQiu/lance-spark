# Managing your Lance Datasets

## CREATE TABLE

```sql
-- Create a simple table
CREATE TABLE users (
    id BIGINT NOT NULL,
    name STRING,
    email STRING,
    created_at TIMESTAMP
);


-- Create table with complex data types
CREATE TABLE events (
    event_id BIGINT NOT NULL,
    user_id BIGINT,
    event_type STRING,
    tags ARRAY<STRING>,
    metadata STRUCT<
        source: STRING,
        version: INT,
        processed_at: TIMESTAMP
    >,
    occurred_at TIMESTAMP
);
```

## INSERT INTO

```sql
-- Insert individual rows
INSERT INTO users VALUES 
    (4, 'David', 'david@example.com', '2024-01-15 10:30:00'),
    (5, 'Eva', 'eva@example.com', '2024-01-15 11:45:00');

-- Insert with column specification
INSERT INTO users (id, name, email) VALUES 
    (6, 'Frank', 'frank@example.com'),
    (7, 'Grace', 'grace@example.com');

-- Insert from SELECT query
INSERT INTO users
SELECT user_id as id, username as name, email_address as email, signup_date as created_at
FROM staging.user_signups
WHERE signup_date >= '2024-01-01';

-- Insert with complex data types
INSERT INTO events VALUES (
    1001,
    123,
    'page_view',
    map('page', '/home', 'referrer', 'google'),
    array('web', 'desktop'),
    struct('web_app', 1, '2024-01-15 12:00:00'),
    '2024-01-15 12:00:00'
);
```

## DROP TABLE

```sql
-- Drop table
DROP TABLE users;

-- Drop table if it exists (no error if table doesn't exist)
DROP TABLE IF EXISTS users;
```

## DataFrame CreateTable

=== "Python"
    ```python
    # Create DataFrame
    data = [
    (1, "Alice", "alice@example.com"),
    (2, "Bob", "bob@example.com"),
    (3, "Charlie", "charlie@example.com")
    ]
    df = spark.createDataFrame(data, ["id", "name", "email"])
    
    # Write as new table using catalog
    df.writeTo("users").create()
    ```

=== "Scala"
    ```scala
    import spark.implicits._
    
    // Create DataFrame
    val data = Seq(
        (1, "Alice", "alice@example.com"),
        (2, "Bob", "bob@example.com"),
        (3, "Charlie", "charlie@example.com")
    )
    val df = data.toDF("id", "name", "email")
    
    // Write as new table using catalog
    df.writeTo("users").create()
    ```

=== "Java"
    ```java
    import org.apache.spark.sql.types.*;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    
    // Create DataFrame
    List<Row> data = Arrays.asList(
        RowFactory.create(1L, "Alice", "alice@example.com"),
        RowFactory.create(2L, "Bob", "bob@example.com"),
        RowFactory.create(3L, "Charlie", "charlie@example.com")
    );
    
    StructType schema = new StructType(new StructField[]{
        new StructField("id", DataTypes.LongType, false, Metadata.empty()),
        new StructField("name", DataTypes.StringType, true, Metadata.empty()),
        new StructField("email", DataTypes.StringType, true, Metadata.empty())
    });
    
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    // Write as new table using catalog
    df.writeTo("users").create();
    ```

## DataFrame Write

=== "Python"
    ```python
    # Create new data
    new_data = [
        (8, "Henry", "henry@example.com"),
        (9, "Ivy", "ivy@example.com")
    ]
    new_df = spark.createDataFrame(new_data, ["id", "name", "email"])
    
    # Append to existing table
    new_df.writeTo("users").append()
    
    # Alternative: use traditional write API with mode
    new_df.write.mode("append").saveAsTable("users")
    ```

=== "Scala"
    ```scala
    // Create new data
    val newData = Seq(
        (8, "Henry", "henry@example.com"),
        (9, "Ivy", "ivy@example.com")
    )
    val newDF = newData.toDF("id", "name", "email")
    
    // Append to existing table
    newDF.writeTo("users").append()
    
    // Alternative: use traditional write API with mode
    newDF.write.mode("append").saveAsTable("users")
    ```

=== "Java"
    ```java
    // Create new data
    List<Row> newData = Arrays.asList(
        RowFactory.create(8L, "Henry", "henry@example.com"),
        RowFactory.create(9L, "Ivy", "ivy@example.com")
    );
    Dataset<Row> newDF = spark.createDataFrame(newData, schema);
    
    // Append to existing table
    newDF.writeTo("users").append();
    
    // Alternative: use traditional write API with mode
    newDF.write().mode("append").saveAsTable("users");
    ```

## Vector Columns (FixedSizeList)

Lance supports efficient storage of vector/embedding columns using Arrow's FixedSizeList type. This is essential for machine learning applications where vectors must have consistent dimensions.

### Writing Vector Columns

To write vector columns, add metadata to your DataFrame schema specifying the fixed size:

=== "Python"
    ```python
    from pyspark.sql import SparkSession
    from pyspark.sql.types import *
    
    # Create metadata for 128-dimensional vectors
    metadata = {"arrow.FixedSizeList.size": 128}
    
    # Define schema with vector column
    schema = StructType([
        StructField("id", LongType(), nullable=False),
        StructField("text", StringType(), nullable=True),
        StructField("embeddings", ArrayType(FloatType()), 
                    nullable=False, metadata=metadata)
    ])
    
    # Create DataFrame with vector data
    data = [
        (1, "first document", [0.1] * 128),
        (2, "second document", [0.2] * 128)
    ]
    
    df = spark.createDataFrame(data, schema)
    
    # Write to Lance - embeddings stored as FixedSizeList[128]
    df.write.format("lance").mode("overwrite").save("vectors.lance")
    ```

=== "Scala"
    ```scala
    import org.apache.spark.sql.types._
    
    // Create metadata for 128-dimensional vectors
    val metadata = new MetadataBuilder()
      .putLong("arrow.FixedSizeList.size", 128)
      .build()
    
    // Define schema with vector column
    val schema = StructType(Seq(
      StructField("id", LongType, nullable = false),
      StructField("text", StringType, nullable = true),
      StructField("embeddings", ArrayType(FloatType), 
                  nullable = false, metadata)
    ))
    
    // Create DataFrame with vector data
    val data = Seq(
      (1L, "first document", Array.fill(128)(0.1f)),
      (2L, "second document", Array.fill(128)(0.2f))
    )
    
    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(data), schema)
    
    // Write to Lance - embeddings stored as FixedSizeList[128]
    df.write.format("lance").mode("overwrite").save("vectors.lance")
    ```

=== "Java"
    ```java
    import org.apache.spark.sql.types.*;
    
    // Create metadata for 128-dimensional vectors
    Metadata metadata = new MetadataBuilder()
        .putLong("arrow.FixedSizeList.size", 128)
        .build();
    
    // Define schema with vector column
    StructType schema = new StructType(new StructField[]{
        new StructField("id", DataTypes.LongType, false, Metadata.empty()),
        new StructField("text", DataTypes.StringType, true, Metadata.empty()),
        new StructField("embeddings", DataTypes.createArrayType(DataTypes.FloatType), 
                        false, metadata)
    });
    
    // Create DataFrame with vector data
    float[] embedding1 = new float[128];
    float[] embedding2 = new float[128];
    Arrays.fill(embedding1, 0.1f);
    Arrays.fill(embedding2, 0.2f);
    
    List<Row> data = Arrays.asList(
        RowFactory.create(1L, "first document", embedding1),
        RowFactory.create(2L, "second document", embedding2)
    );
    
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    // Write to Lance - embeddings stored as FixedSizeList[128]
    df.write().format("lance").mode("overwrite").save("vectors.lance");
    ```

### Benefits of FixedSizeList

1. **Type Safety**: Ensures all vectors have the same dimension
2. **Performance**: Better memory layout and SIMD operations
3. **Index Support**: Required for Lance vector indexing and similarity search
4. **Storage Efficiency**: More efficient columnar compression

### Validation

The connector validates vector columns at write time:

1. **Null Check**: Vector columns cannot contain null values
   ```
   IllegalArgumentException: Vector column 'embeddings' cannot be null at row position 0
   ```

2. **Dimension Check**: All vectors must have the exact specified dimension
   ```
   IllegalArgumentException: Vector column 'embeddings' expected dimension 128 but got 64 at row position 5
   ```

### Supported Types

FixedSizeList conversion works with:
- `FloatType` (float32 - most common for embeddings)
- `DoubleType` (float64 - higher precision vectors)

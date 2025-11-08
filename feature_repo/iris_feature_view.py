from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, String

# Define data source (Parquet recommended)
iris_source = FileSource(
    path="../ga_resources/data/raw/iris_feast.parquet",
    timestamp_field="event_timestamp",
)

# Define entity (primary key)
iris_entity = Entity(
    name="flower_id",
    join_keys=["flower_id"],
    value_type=ValueType.INT64,
    description="Unique identifier for each flower sample",
)

# Define feature view
iris_fv = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=None,
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="species", dtype=String),
    ],
    online=True,
    source=iris_source,
)
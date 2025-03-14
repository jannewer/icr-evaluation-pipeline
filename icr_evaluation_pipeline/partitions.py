from dagster import StaticPartitionsDefinition

dataset_partitions = StaticPartitionsDefinition(
    ["Iris", "Ecoli", "Glass Identification", "Haberman's Survival"]
)

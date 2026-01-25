from pydantic import BaseModel, ConfigDict, Field, field_serializer
from typing import Any


class zenml_parse:
    def __init__(
        self,
        model_name: str = "None",
        model_class: str = "None",
        y_variable: str = "None",
        dropped_columns: list = [],
        transformations: str = "None",
        outliers: bool = True,
        file_trained_on: str = "None",
        random_state: int = 0,
        uri : str = "None",
        run_name : str = "None",
        registered_model_name: str = "None"
    ) -> None:
        self.model_name = model_name
        self.model_class = model_class
        self.y_variable = y_variable
        self.dropped_columns = dropped_columns
        self.transformations = transformations
        self.outliers = outliers
        self.data_path = file_trained_on
        self.random_state = random_state
        self.uri = uri
        self.run_name = run_name
        self.registered_model_name = registered_model_name

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "model_class": self.model_class,
            "y_variable": self.y_variable,
            "dropped_columns": self.dropped_columns,
            "transformations": self.transformations,
            "outliers": self.outliers,
            "data_path": self.data_path,
            "random_state": self.random_state,
            "uri": self.uri,
            "run_name": self.run_name,
            "registered_model_name": self.registered_model_name
        }


def default_zenml_parse():
    return zenml_parse(
        model_name="default",
        model_class="default",
        y_variable="default",
        dropped_columns=[],
        transformations="none",
        outliers=False,
        file_trained_on="",
        random_state=42,
        uri = "None",
        run_name= "None",
        registered_model_name= "None"
    )


class pydantic_model(BaseModel):
    zenml_data: zenml_parse 
    # = Field(default_factory=default_zenml_parse)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("zenml_data")
    def serialize_zenml_data(self, zenml_data: zenml_parse, _info) -> dict[str, Any]:
        return zenml_data.to_dict()

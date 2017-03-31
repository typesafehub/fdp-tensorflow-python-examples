node {
  name: "x"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "weight"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 0.345968902111
      }
    }
  }
}
node {
  name: "bias"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: -0.0314258821309
      }
    }
  }
}
node {
  name: "Mul"
  op: "Mul"
  input: "x"
  input: "weight"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Y_pred"
  op: "Add"
  input: "Mul"
  input: "bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}

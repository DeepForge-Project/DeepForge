module {
  func.func @deepforge_conv2d(%x: tensor<1x5x5x3xf32>,
                              %w: tensor<5x3x3x3xf32>,
                              %y_init: tensor<1x5x5x5xf32>) -> tensor<1x5x5x5xf32> {
    %zero = arith.constant 0.0 : f32
    %padded = tensor.pad %x low[0, 1, 0, 0] high[0, 1, 2, 0] {
    ^bb0(%n: index, %h: index, %ww: index, %c: index):
      tensor.yield %zero : f32
    } : tensor<1x5x5x3xf32> to tensor<1x7x7x3xf32>
    %filled = linalg.fill ins(%zero : f32)
        outs(%y_init : tensor<1x5x5x5xf32>) -> tensor<1x5x5x5xf32>
    %result = linalg.conv_2d_nhwc_fhwc
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%padded, %w : tensor<1x7x7x3xf32>, tensor<5x3x3x3xf32>)
        outs(%filled : tensor<1x5x5x5xf32>) -> tensor<1x5x5x5xf32>
    return %result : tensor<1x5x5x5xf32>
  }
}

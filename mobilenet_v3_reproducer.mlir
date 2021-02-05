"module"() ( {
  "func"() ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<1x1x16x16xf32>, %arg2: tensor<1x1x16x16xf32>, %arg3: tensor<3x3x1x16xf32>, %arg4: tensor<8xf32>, %arg5: tensor<1x1x16x8xf32>, %arg6: tensor<16xf32>, %arg7: tensor<1x1x8x16xf32>, %arg8: tensor<1x16x16x16xf32>, %arg9: tensor<1xi32>):  // no predecessors
    %0 = "mhlo.constant"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %1 = "mhlo.constant"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %2 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x16x16x16xf32>
    %4 = "mhlo.constant"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %5 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x16x16x16xf32>
    %7 = "mhlo.constant"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %8 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %9 = "mhlo.broadcast_in_dim"(%8) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x8xf32>
    %10 = "mhlo.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %11 = "mhlo.add"(%arg0, %10) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %12 = "mhlo.convolution"(%arg8, %arg1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x16x16x16xf32>, tensor<1x1x16x16xf32>) -> tensor<1x16x16x16xf32>
    %13 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %14 = "mhlo.broadcast_in_dim"(%13) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x16x16x16xf32>
    %15 = "mhlo.compare"(%12, %14) {comparison_direction = "GT"} : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xi1>
    %16 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %17 = "mhlo.broadcast_in_dim"(%16) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x16x16x16xf32>
    %18 = "mhlo.maximum"(%12, %17) : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %19 = "mhlo.tuple"(%18) : (tensor<1x16x16x16xf32>) -> tuple<tensor<1x16x16x16xf32>>
    %20 = "mhlo.get_tuple_element"(%19) {index = 0 : i32} : (tuple<tensor<1x16x16x16xf32>>) -> tensor<1x16x16x16xf32>
    %21 = "mhlo.convolution"(%20, %arg3) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 16 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<1> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x16x16x16xf32>, tensor<3x3x1x16xf32>) -> tensor<1x16x16x16xf32>
    %22 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %23 = "mhlo.broadcast_in_dim"(%22) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x16x16x16xf32>
    %24 = "mhlo.compare"(%21, %23) {comparison_direction = "GT"} : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xi1>
    %25 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %26 = "mhlo.broadcast_in_dim"(%25) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x16x16x16xf32>
    %27 = "mhlo.maximum"(%21, %26) : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %28 = "mhlo.tuple"(%27) : (tensor<1x16x16x16xf32>) -> tuple<tensor<1x16x16x16xf32>>
    %29 = "mhlo.get_tuple_element"(%28) {index = 0 : i32} : (tuple<tensor<1x16x16x16xf32>>) -> tensor<1x16x16x16xf32>
    %30 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %31 = "mhlo.reduce"(%29, %30) ( {
    ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):  // no predecessors
      %208 = "mhlo.add"(%arg10, %arg11) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x16x16x16xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %32 = "mhlo.constant"() {value = dense<16> : tensor<i32>} : () -> tensor<i32>
    %33 = "mhlo.convert"(%32) : (tensor<i32>) -> tensor<f32>
    %34 = "mhlo.broadcast_in_dim"(%33) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x16x16xf32>
    %35 = "mhlo.divide"(%31, %34) : (tensor<1x16x16xf32>, tensor<1x16x16xf32>) -> tensor<1x16x16xf32>
    %36 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %37 = "mhlo.reduce"(%35, %36) ( {
    ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):  // no predecessors
      %208 = "mhlo.add"(%arg10, %arg11) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<1x16xf32>
    %38 = "mhlo.constant"() {value = dense<16> : tensor<i32>} : () -> tensor<i32>
    %39 = "mhlo.convert"(%38) : (tensor<i32>) -> tensor<f32>
    %40 = "mhlo.broadcast_in_dim"(%39) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x16xf32>
    %41 = "mhlo.divide"(%37, %40) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %42 = "mhlo.reshape"(%41) : (tensor<1x16xf32>) -> tensor<1x1x1x16xf32>
    %43 = "mhlo.convolution"(%42, %arg5) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x16xf32>, tensor<1x1x16x8xf32>) -> tensor<1x1x1x8xf32>
    %44 = "mhlo.broadcast_in_dim"(%arg4) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>) -> tensor<1x1x1x8xf32>
    %45 = "mhlo.add"(%43, %44) : (tensor<1x1x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
    %46 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %47 = "mhlo.broadcast_in_dim"(%46) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x8xf32>
    %48 = "mhlo.compare"(%45, %47) {comparison_direction = "GT"} : (tensor<1x1x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xi1>
    %49 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %50 = "mhlo.negate"(%49) : (tensor<f32>) -> tensor<f32>
    %51 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %52 = "mhlo.divide"(%50, %51) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %53 = "mhlo.broadcast_in_dim"(%52) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1xf32>
    %54 = "mhlo.broadcast_in_dim"(%53) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x4096xf32>
    %55 = "mhlo.convert"(%arg9) : (tensor<1xi32>) -> tensor<1xf32>
    %56 = "mhlo.broadcast_in_dim"(%55) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x1xf32>
    %57 = "mhlo.reshape"(%56) : (tensor<1x1xf32>) -> tensor<1xf32>
    %58 = "mhlo.broadcast_in_dim"(%57) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x4096xf32>
    %59 = "mhlo.multiply"(%54, %58) : (tensor<1x4096xf32>, tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %60 = "mhlo.reshape"(%59) : (tensor<1x4096xf32>) -> tensor<1x16x16x16xf32>
    %61 = "mhlo.reverse"(%arg2) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
    %62 = "mhlo.convolution"(%60, %61) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 3 : i64, kernel_output_feature_dimension = 2 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x16x16x16xf32>, tensor<1x1x16x16xf32>) -> tensor<1x16x16x16xf32>
    %63 = "mhlo.multiply"(%62, %29) : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %64 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %65 = "mhlo.reduce"(%63, %64) ( {
    ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):  // no predecessors
      %208 = "mhlo.add"(%arg10, %arg11) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %66 = "mhlo.broadcast_in_dim"(%65) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16xf32>) -> tensor<1x16xf32>
    %67 = "mhlo.broadcast_in_dim"(%66) {broadcast_dimensions = dense<[0, 3]> : tensor<2xi64>} : (tensor<1x16xf32>) -> tensor<1x1x1x16xf32>
    %68 = "mhlo.constant"() {value = dense<6.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %69 = "mhlo.broadcast_in_dim"(%68) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %70 = "mhlo.divide"(%67, %69) : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %71 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %72 = "mhlo.broadcast_in_dim"(%71) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x8xf32>
    %73 = "mhlo.maximum"(%45, %72) : (tensor<1x1x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
    %74 = "mhlo.tuple"(%73) : (tensor<1x1x1x8xf32>) -> tuple<tensor<1x1x1x8xf32>>
    %75 = "mhlo.get_tuple_element"(%74) {index = 0 : i32} : (tuple<tensor<1x1x1x8xf32>>) -> tensor<1x1x1x8xf32>
    %76 = "mhlo.convolution"(%75, %arg7) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x8xf32>, tensor<1x1x8x16xf32>) -> tensor<1x1x1x16xf32>
    %77 = "mhlo.broadcast_in_dim"(%arg6) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %78 = "mhlo.add"(%76, %77) : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %79 = "mhlo.constant"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %80 = "mhlo.broadcast_in_dim"(%79) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %81 = "mhlo.add"(%78, %80) : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %82 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %83 = "mhlo.broadcast_in_dim"(%82) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %84 = "mhlo.maximum"(%81, %83) : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %85 = "mhlo.constant"() {value = dense<6.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %86 = "mhlo.broadcast_in_dim"(%85) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %87 = "mhlo.minimum"(%84, %86) : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %88 = "mhlo.compare"(%84, %87) {comparison_direction = "EQ"} : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xi1>
    %89 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %90 = "mhlo.broadcast_in_dim"(%89) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %91 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %92 = "mhlo.broadcast_in_dim"(%91) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %93 = "mhlo.select"(%88, %90, %92) : (tensor<1x1x1x16xi1>, tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %94 = "mhlo.constant"() {value = dense<6.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %95 = "mhlo.broadcast_in_dim"(%94) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %96 = "mhlo.compare"(%95, %87) {comparison_direction = "EQ"} : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xi1>
    %97 = "mhlo.constant"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %98 = "mhlo.broadcast_in_dim"(%97) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %99 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %100 = "mhlo.broadcast_in_dim"(%99) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %101 = "mhlo.select"(%96, %98, %100) : (tensor<1x1x1x16xi1>, tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %102 = "mhlo.divide"(%93, %101) : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %103 = "mhlo.multiply"(%70, %102) : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %104 = "mhlo.compare"(%81, %84) {comparison_direction = "EQ"} : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xi1>
    %105 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %106 = "mhlo.broadcast_in_dim"(%105) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %107 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %108 = "mhlo.broadcast_in_dim"(%107) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %109 = "mhlo.select"(%104, %106, %108) : (tensor<1x1x1x16xi1>, tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %110 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %111 = "mhlo.broadcast_in_dim"(%110) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %112 = "mhlo.compare"(%111, %84) {comparison_direction = "EQ"} : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xi1>
    %113 = "mhlo.constant"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %114 = "mhlo.broadcast_in_dim"(%113) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %115 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %116 = "mhlo.broadcast_in_dim"(%115) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %117 = "mhlo.select"(%112, %114, %116) : (tensor<1x1x1x16xi1>, tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %118 = "mhlo.divide"(%109, %117) : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %119 = "mhlo.multiply"(%103, %118) : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %120 = "mhlo.reverse"(%arg7) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1x8x16xf32>) -> tensor<1x1x8x16xf32>
    %121 = "mhlo.convolution"(%119, %120) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 3 : i64, kernel_output_feature_dimension = 2 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x16xf32>, tensor<1x1x8x16xf32>) -> tensor<1x1x1x8xf32>
    %122 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %123 = "mhlo.broadcast_in_dim"(%122) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x8xf32>
    %124 = "mhlo.select"(%48, %121, %123) : (tensor<1x1x1x8xi1>, tensor<1x1x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
    %125 = "mhlo.reverse"(%arg5) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1x16x8xf32>) -> tensor<1x1x16x8xf32>
    %126 = "mhlo.convolution"(%124, %125) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 3 : i64, kernel_output_feature_dimension = 2 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x1x1x16xf32>
    %127 = "mhlo.reshape"(%126) : (tensor<1x1x1x16xf32>) -> tensor<1x16xf32>
    %128 = "mhlo.broadcast_in_dim"(%39) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x16xf32>
    %129 = "mhlo.divide"(%127, %128) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %130 = "mhlo.broadcast_in_dim"(%129) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x16xf32>) -> tensor<1x16x16xf32>
    %131 = "mhlo.broadcast_in_dim"(%33) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x16x16xf32>
    %132 = "mhlo.divide"(%130, %131) : (tensor<1x16x16xf32>, tensor<1x16x16xf32>) -> tensor<1x16x16xf32>
    %133 = "mhlo.broadcast_in_dim"(%132) {broadcast_dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x16x16xf32>) -> tensor<1x16x16x16xf32>
    %134 = "mhlo.constant"() {value = dense<6.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %135 = "mhlo.broadcast_in_dim"(%134) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x1x16xf32>
    %136 = "mhlo.divide"(%87, %135) : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %137 = "mhlo.reshape"(%136) : (tensor<1x1x1x16xf32>) -> tensor<1x16xf32>
    %138 = "mhlo.broadcast_in_dim"(%137) {broadcast_dimensions = dense<[0, 3]> : tensor<2xi64>} : (tensor<1x16xf32>) -> tensor<1x16x16x16xf32>
    %139 = "mhlo.multiply"(%62, %138) : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %140 = "mhlo.add"(%133, %139) : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %141 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %142 = "mhlo.broadcast_in_dim"(%141) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x16x16x16xf32>
    %143 = "mhlo.select"(%24, %140, %142) : (tensor<1x16x16x16xi1>, tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %144 = "mhlo.reshape"(%arg3) : (tensor<3x3x1x16xf32>) -> tensor<3x3x1x16x1xf32>
    %145 = "mhlo.transpose"(%144) {minor_to_major = dense<[4, 2, 3, 1, 0]> : tensor<5xindex>, permutation = dense<[0, 1, 3, 2, 4]> : tensor<5xi64>} : (tensor<3x3x1x16x1xf32>) -> tensor<3x3x16x1x1xf32>
    %146 = "mhlo.reshape"(%145) : (tensor<3x3x16x1x1xf32>) -> tensor<3x3x16x1xf32>
    %147 = "mhlo.reverse"(%146) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<3x3x16x1xf32>) -> tensor<3x3x16x1xf32>
    %148 = "mhlo.convolution"(%143, %147) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 3 : i64, kernel_output_feature_dimension = 2 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 16 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<1> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x16x16x16xf32>, tensor<3x3x16x1xf32>) -> tensor<1x16x16x16xf32>
    %149 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %150 = "mhlo.broadcast_in_dim"(%149) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x16x16x16xf32>
    %151 = "mhlo.select"(%15, %148, %150) : (tensor<1x16x16x16xi1>, tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %152 = "mhlo.convolution"(%arg8, %151) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 3 : i64, input_feature_dimension = 0 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 0 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, output_batch_dimension = 2 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x1x16x16xf32>
    %153 = "mhlo.constant"() {value = dense<1.000000e-03> : tensor<f32>} : () -> tensor<f32>
    %154 = "mhlo.broadcast_in_dim"(%153) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x16x16xf32>
    %155 = "mhlo.multiply"(%152, %154) : (tensor<1x1x16x16xf32>, tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
    %156 = "mhlo.subtract"(%arg1, %155) : (tensor<1x1x16x16xf32>, tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
    %157 = "mhlo.reshape"(%136) : (tensor<1x1x1x16xf32>) -> tensor<1x16xf32>
    %158 = "mhlo.broadcast_in_dim"(%157) {broadcast_dimensions = dense<[0, 3]> : tensor<2xi64>} : (tensor<1x16xf32>) -> tensor<1x16x16x16xf32>
    %159 = "mhlo.multiply"(%29, %158) : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %160 = "mhlo.convolution"(%159, %60) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 3 : i64, input_feature_dimension = 0 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 0 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, output_batch_dimension = 2 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x1x16x16xf32>
    %161 = "mhlo.constant"() {value = dense<1.000000e-03> : tensor<f32>} : () -> tensor<f32>
    %162 = "mhlo.broadcast_in_dim"(%161) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x16x16xf32>
    %163 = "mhlo.multiply"(%160, %162) : (tensor<1x1x16x16xf32>, tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
    %164 = "mhlo.subtract"(%arg2, %163) : (tensor<1x1x16x16xf32>, tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
    %165 = "mhlo.convolution"(%20, %143) {batch_group_count = 16 : i64, dimension_numbers = {input_batch_dimension = 3 : i64, input_feature_dimension = 0 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 0 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, output_batch_dimension = 2 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<1> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<3x3x1x16xf32>
    %166 = "mhlo.constant"() {value = dense<1.000000e-03> : tensor<f32>} : () -> tensor<f32>
    %167 = "mhlo.broadcast_in_dim"(%166) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<3x3x1x16xf32>
    %168 = "mhlo.multiply"(%165, %167) : (tensor<3x3x1x16xf32>, tensor<3x3x1x16xf32>) -> tensor<3x3x1x16xf32>
    %169 = "mhlo.subtract"(%arg3, %168) : (tensor<3x3x1x16xf32>, tensor<3x3x1x16xf32>) -> tensor<3x3x1x16xf32>
    %170 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %171 = "mhlo.reduce"(%124, %170) ( {
    ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):  // no predecessors
      %208 = "mhlo.add"(%arg10, %arg11) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x1x8xf32>, tensor<f32>) -> tensor<8xf32>
    %172 = "mhlo.broadcast_in_dim"(%171) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<8xf32>) -> tensor<8xf32>
    %173 = "mhlo.constant"() {value = dense<1.000000e-03> : tensor<f32>} : () -> tensor<f32>
    %174 = "mhlo.broadcast_in_dim"(%173) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<8xf32>
    %175 = "mhlo.multiply"(%172, %174) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %176 = "mhlo.subtract"(%arg4, %175) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %177 = "mhlo.convolution"(%42, %124) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 3 : i64, input_feature_dimension = 0 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 0 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, output_batch_dimension = 2 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x16xf32>, tensor<1x1x1x8xf32>) -> tensor<1x1x16x8xf32>
    %178 = "mhlo.constant"() {value = dense<1.000000e-03> : tensor<f32>} : () -> tensor<f32>
    %179 = "mhlo.broadcast_in_dim"(%178) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x16x8xf32>
    %180 = "mhlo.multiply"(%177, %179) : (tensor<1x1x16x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x1x16x8xf32>
    %181 = "mhlo.subtract"(%arg5, %180) : (tensor<1x1x16x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x1x16x8xf32>
    %182 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %183 = "mhlo.reduce"(%119, %182) ( {
    ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):  // no predecessors
      %208 = "mhlo.add"(%arg10, %arg11) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x1x16xf32>, tensor<f32>) -> tensor<16xf32>
    %184 = "mhlo.broadcast_in_dim"(%183) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<16xf32>) -> tensor<16xf32>
    %185 = "mhlo.constant"() {value = dense<1.000000e-03> : tensor<f32>} : () -> tensor<f32>
    %186 = "mhlo.broadcast_in_dim"(%185) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16xf32>
    %187 = "mhlo.multiply"(%184, %186) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %188 = "mhlo.subtract"(%arg6, %187) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %189 = "mhlo.convolution"(%75, %119) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 3 : i64, input_feature_dimension = 0 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 0 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, output_batch_dimension = 2 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x8xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x8x16xf32>
    %190 = "mhlo.constant"() {value = dense<1.000000e-03> : tensor<f32>} : () -> tensor<f32>
    %191 = "mhlo.broadcast_in_dim"(%190) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x8x16xf32>
    %192 = "mhlo.multiply"(%189, %191) : (tensor<1x1x8x16xf32>, tensor<1x1x8x16xf32>) -> tensor<1x1x8x16xf32>
    %193 = "mhlo.subtract"(%arg7, %192) : (tensor<1x1x8x16xf32>, tensor<1x1x8x16xf32>) -> tensor<1x1x8x16xf32>
    %194 = "mhlo.convolution"(%159, %arg2) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x16x16x16xf32>, tensor<1x1x16x16xf32>) -> tensor<1x16x16x16xf32>
    %195 = "mhlo.add"(%arg8, %194) : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %196 = "mhlo.reshape"(%195) : (tensor<1x16x16x16xf32>) -> tensor<1x4096xf32>
    %197 = "mhlo.reshape"(%56) : (tensor<1x1xf32>) -> tensor<1xf32>
    %198 = "mhlo.broadcast_in_dim"(%197) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x4096xf32>
    %199 = "mhlo.multiply"(%196, %198) : (tensor<1x4096xf32>, tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %200 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %201 = "mhlo.reduce"(%199, %200) ( {
    ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):  // no predecessors
      %208 = "mhlo.add"(%arg10, %arg11) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
    %202 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %203 = "mhlo.reduce"(%201, %202) ( {
    ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):  // no predecessors
      %208 = "mhlo.add"(%arg10, %arg11) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>, tensor<f32>) -> tensor<f32>
    %204 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %205 = "mhlo.divide"(%203, %204) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %206 = "mhlo.negate"(%205) : (tensor<f32>) -> tensor<f32>
    %207 = "mhlo.tuple"(%11, %156, %164, %169, %176, %181, %188, %193, %206) : (tensor<i32>, tensor<1x1x16x16xf32>, tensor<1x1x16x16xf32>, tensor<3x3x1x16xf32>, tensor<8xf32>, tensor<1x1x16x8xf32>, tensor<16xf32>, tensor<1x1x8x16xf32>, tensor<f32>) -> tuple<tensor<i32>, tensor<1x1x16x16xf32>, tensor<1x1x16x16xf32>, tensor<3x3x1x16xf32>, tensor<8xf32>, tensor<1x1x16x8xf32>, tensor<16xf32>, tensor<1x1x8x16xf32>, tensor<f32>>
    "std.return"(%207) : (tuple<tensor<i32>, tensor<1x1x16x16xf32>, tensor<1x1x16x16xf32>, tensor<3x3x1x16xf32>, tensor<8xf32>, tensor<1x1x16x8xf32>, tensor<16xf32>, tensor<1x1x8x16xf32>, tensor<f32>>) -> ()
  }) {iree.module.export, sym_name = "main", type = (tensor<i32>, tensor<1x1x16x16xf32>, tensor<1x1x16x16xf32>, tensor<3x3x1x16xf32>, tensor<8xf32>, tensor<1x1x16x8xf32>, tensor<16xf32>, tensor<1x1x8x16xf32>, tensor<1x16x16x16xf32>, tensor<1xi32>) -> tuple<tensor<i32>, tensor<1x1x16x16xf32>, tensor<1x1x16x16xf32>, tensor<3x3x1x16xf32>, tensor<8xf32>, tensor<1x1x16x8xf32>, tensor<16xf32>, tensor<1x1x8x16xf32>, tensor<f32>>} : () -> ()
  "module_terminator"() : () -> ()
}) : () -> ()

import vtkmodules.all as vtk
import vtkmodules.util.numpy_support as numpy_support 

def ndarray2structured_point(data, output_name="output") -> None:
    vtk_array = numpy_support.numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

    ## Initialize VtkStrcturePoints ##
    structured_points = vtk.vtkStructuredPoints()
    structured_points.SetDimensions(data.shape[0], data.shape[1], data.shape[2])  # 设置vtk数据的维度
    structured_points.SetOrigin(0, 0, 0)  # 设置vtk数据的原点位置
    structured_points.SetSpacing(1, 1, 1)  # 设置vtk数据的间距
    structured_points.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(output_name+".vtk")
    writer.SetInputData(structured_points)
    writer.Write() 


if __name__ == "__main__":
    pass
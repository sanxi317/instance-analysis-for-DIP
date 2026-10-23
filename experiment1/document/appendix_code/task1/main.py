import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pathlib as path

basic_path = path.Path("/home/sanxi/DIP/experiment1")
patient_ID="L067"

full=basic_path / "full_3mm" / patient_ID / "full_3mm" / "L067_FD_3_1.CT.0002.0001.2015.12.22.18.12.07.5968.358090401.IMA"
quarter=basic_path / "quarter_3mm" / patient_ID / "quarter_3mm" / "L067_QD_3_1.CT.0004.0001.2015.12.22.18.12.56.428910.358293547.IMA"
#取出图像的中间切片
output_dir=basic_path / "output" 
output_dir.mkdir(parents=True, exist_ok=True)

ct_image=sitk.ReadImage(str(full))
print(ct_image.GetSize())
ct_array=sitk.GetArrayFromImage(ct_image)
print(ct_array.shape)
ct_slice=ct_array[0]
print(ct_slice.shape)
#将中间切片保存为png格式
plt.imshow(
    ct_slice,
    cmap="gray",
    vmin=-1350,
    vmax=150,
    interpolation="nearest"
)
plt.title("CT Slice")
plt.axis('off')
save_path=output_dir / "ct_slice_full.png"#保存
plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()



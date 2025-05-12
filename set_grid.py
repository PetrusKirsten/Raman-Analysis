import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def montar_grid_topografia(
        title, image_paths, save_path,
        crop_list: list, cols=4, rows=3, row_labels=["St CL", "St kCar CL", "St iCar CL"]):
    # === Fonte ===
    font_path = (
        "C:/Users/petru/AppData/Local/Programs/Python/Python313/Lib/site-packages/matplotlib/"
        "mpl-data/fonts/ttf/helvetica-light-587ebe5a59211.ttf")
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    font_name = prop.get_name()

    plt.rcParams.update({
        'font.family': font_name,
        'figure.facecolor': '#09141E',
        'axes.facecolor': '#09141E',
        'savefig.dpi': 300,
    })


    # === Recorte ===
    crop_x, crop_y, crop_w, crop_h = crop_list[0], crop_list[1], crop_list[2], crop_list[3]

    col_labels = [f"CL {i * 7}" for i in range(cols)]

    cropped_images = []
    for path in image_paths:
        img = Image.open(path)
        crop_box = (
            crop_x,
            crop_y,
            min(crop_x + crop_w, img.width),
            min(crop_y + crop_h, img.height)
        )
        cropped = img.crop(crop_box)
        cropped_images.append(cropped)

    img_w, img_h = cropped_images[0].size
    final_img = Image.new("RGB", (img_w * cols, img_h * rows))

    for idx, img in enumerate(cropped_images):
        row = idx // cols
        col = idx % cols
        final_img.paste(img, (col * img_w, row * img_h))

    # === Plotagem com título e labels ===
    fig, ax = plt.subplots(figsize=(cols * 3, rows * 3), constrained_layout=True)
    ax.imshow(final_img)
    ax.axis("off")

    ax.set_xlim(-100, final_img.width)
    ax.set_ylim(final_img.height, -100)

    plt.suptitle(title, fontsize=16, color='white', fontproperties=prop)

    for c in range(cols):
        ax.text((c + 0.5) * img_w, -30, col_labels[c], ha='center', va='bottom',
                fontsize=12, color='w', fontproperties=prop)

    for r in range(rows):
        ax.text(-30, (r + 0.5) * img_h, row_labels[r], ha='right', va='center',
                fontsize=12, color='w', fontproperties=prop, rotation=90)

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()


# # === Topograhpy ===
# type = 'topography'
#
# st = f"./figures/maps/St CLs/local/topography_40to1785_w-bg_nearest"
# kc = f"./figures/maps/St kC CLs/local/topography_40to1785_w-bg_nearest"
# ic = f"./figures/maps/St iC CLs/local/topography_40to1785_w-bg_nearest"
#
# paths_topo = [
#         f"{st}/St_CL_0_Region_1_{type}.png",
#         f"{st}/St_CL_7_Region_2_{type}.png",
#         f"{st}/St_CL_14_Region_2_{type}.png",
#         f"{st}/St_CL_21_Region_1_{type}.png",
#
#         f"{kc}/St_kC_CL_0_Region_1_{type}.png",
#         f"{kc}/St_kC_CL_7_Region_1_{type}.png",
#         f"{kc}/St_kC_CL_14_Region_2_{type}.png",
#         f"{kc}/St_kC_CL_21_Region_1_{type}.png",
#
#         f"{ic}/St_iC_CL_0_Region_1_{type}.png",
#         f"{ic}/St_iC_CL_7_Region_2_{type}.png",
#         f"{ic}/St_iC_CL_14_Region_1_{type}.png",
#         f"{ic}/St_iC_CL_21_Region_3_{type}.png",
# ]
#
# montar_grid_topografia(
#     title=f"Total spectrum sum | Topography map", image_paths=paths_topo, save_path=f"{type}_grid.png",
#     crop_list=[0, 240, 2160, 2100]
# )
#
# # === 480 cm-1 band ===
# type = 'band_480_diff_global'
# band = '480'
#
# st = f"./figures/maps/St CLs/bands_40to1785_w-bg_nearest"
# kc = f"./figures/maps/St kC CLs/bands_40to1785_w-bg_nearest"
# ic = f"./figures/maps/St iC CLs/bands_40to1785_w-bg_nearest"
#
# paths_band = [
#         f"{st}/{band}/St_CL_0_Region_1_{type}.png",
#         f"{st}/{band}/St_CL_7_Region_2_{type}.png",
#         f"{st}/{band}/St_CL_14_Region_2_{type}.png",
#         f"{st}/{band}/St_CL_21_Region_1_{type}.png",
#
#         f"{kc}/{band}/St_kC_CL_0_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_2_{type}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_2_{type}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_1_{type}.png",
#
#         f"{ic}/{band}/St_iC_CL_0_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_2_{type}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_2_{type}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_1_{type}.png",
# ]
#
# montar_grid_topografia(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{type}_grid.png",
#     crop_list=[0, 300, 2100, 2000]
# )
#
# # === 480 cm-1 band - raw ===
# type = 'band_480_raw_global'
# band = '480'
#
# st = f"./figures/maps/St CLs/bands_40to1785_w-bg_nearest"
# kc = f"./figures/maps/St kC CLs/bands_40to1785_w-bg_nearest"
# ic = f"./figures/maps/St iC CLs/bands_40to1785_w-bg_nearest"
#
# paths_band = [
#         f"{st}/{band}/St_CL_0_Region_1_{type}.png",
#         f"{st}/{band}/St_CL_7_Region_2_{type}.png",
#         f"{st}/{band}/St_CL_14_Region_2_{type}.png",
#         f"{st}/{band}/St_CL_21_Region_1_{type}.png",
#
#         f"{kc}/{band}/St_kC_CL_0_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_2_{type}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_2_{type}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_1_{type}.png",
#
#         f"{ic}/{band}/St_iC_CL_0_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_2_{type}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_2_{type}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_1_{type}.png",
# ]
#
# montar_grid_topografia(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{type}_grid.png",
#     crop_list=[0, 300, 2100, 2000]
# )
#
# # === 550 cm-1 band ===
# band = '550'
# type = f'band_{band}_diff_global'
#
# st = f"./figures/maps/St CLs/bands_40to1785_w-bg_nearest"
# kc = f"./figures/maps/St kC CLs/bands_40to1785_w-bg_nearest"
# ic = f"./figures/maps/St iC CLs/bands_40to1785_w-bg_nearest"
#
# paths_band = [
#         f"{st}/{band}/St_CL_0_Region_1_{type}.png",
#         f"{st}/{band}/St_CL_7_Region_2_{type}.png",
#         f"{st}/{band}/St_CL_14_Region_2_{type}.png",
#         f"{st}/{band}/St_CL_21_Region_2_{type}.png",
#
#         f"{kc}/{band}/St_kC_CL_0_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_2_{type}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_2_{type}.png",
#
#         f"{ic}/{band}/St_iC_CL_0_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_1_{type}.png",
# ]
#
# montar_grid_topografia(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{type}_grid.png",
#     crop_list=[0, 300, 2100, 2000]
# )
#
# # === 550 cm-1 band - raw ===
# band = '550'
# type = f'band_{band}_raw_global'
#
# st = f"./figures/maps/St CLs/bands_40to1785_w-bg_nearest"
# kc = f"./figures/maps/St kC CLs/bands_40to1785_w-bg_nearest"
# ic = f"./figures/maps/St iC CLs/bands_40to1785_w-bg_nearest"
#
# paths_band = [
#         f"{st}/{band}/St_CL_0_Region_1_{type}.png",
#         f"{st}/{band}/St_CL_7_Region_2_{type}.png",
#         f"{st}/{band}/St_CL_14_Region_2_{type}.png",
#         f"{st}/{band}/St_CL_21_Region_2_{type}.png",
#
#         f"{kc}/{band}/St_kC_CL_0_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_2_{type}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_2_{type}.png",
#
#         f"{ic}/{band}/St_iC_CL_0_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_1_{type}.png",
# ]
#
# montar_grid_topografia(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{type}_grid.png",
#     crop_list=[0, 300, 2100, 2000]
# )
#
# # === 941 cm-1 band ===
# band = '941'
# type = f'band_{band}_diff_global'
#
# st = f"./figures/maps/St CLs/bands_40to1785_w-bg_nearest"
# kc = f"./figures/maps/St kC CLs/bands_40to1785_w-bg_nearest"
# ic = f"./figures/maps/St iC CLs/bands_40to1785_w-bg_nearest"
#
# paths_band = [
#         f"{st}/{band}/St_CL_0_Region_1_{type}.png",
#         f"{st}/{band}/St_CL_7_Region_2_{type}.png",
#         f"{st}/{band}/St_CL_14_Region_2_{type}.png",
#         f"{st}/{band}/St_CL_21_Region_1_{type}.png",
#
#         f"{kc}/{band}/St_kC_CL_0_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_2_{type}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_2_{type}.png",
#
#         f"{ic}/{band}/St_iC_CL_0_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_2_{type}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_1_{type}.png",
# ]
#
# montar_grid_topografia(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{type}_grid.png",
#     crop_list=[0, 300, 2100, 2000]
# )
#
# # === 941 cm-1 band - raw ===
# band = '941'
# type = f'band_{band}_raw_global'
#
# st = f"./figures/maps/St CLs/bands_40to1785_w-bg_nearest"
# kc = f"./figures/maps/St kC CLs/bands_40to1785_w-bg_nearest"
# ic = f"./figures/maps/St iC CLs/bands_40to1785_w-bg_nearest"
#
# paths_band = [
#         f"{st}/{band}/St_CL_0_Region_1_{type}.png",
#         f"{st}/{band}/St_CL_7_Region_2_{type}.png",
#         f"{st}/{band}/St_CL_14_Region_2_{type}.png",
#         f"{st}/{band}/St_CL_21_Region_1_{type}.png",
#
#         f"{kc}/{band}/St_kC_CL_0_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_2_{type}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_2_{type}.png",
#
#         f"{ic}/{band}/St_iC_CL_0_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_2_{type}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_1_{type}.png",
# ]
#
# montar_grid_topografia(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{type}_grid.png",
#     crop_list=[0, 300, 2100, 2000]
# )
#
# # === 1220 cm-1 band ===
# band = '1220'
# type = f'band_{band}_diff_global'
#
# kc = f"./figures/maps/St kC CLs/bands_40to1785_w-bg_nearest"
# ic = f"./figures/maps/St iC CLs/bands_40to1785_w-bg_nearest"
#
# paths_band = [
#         f"{kc}/{band}/St_kC_CL_0_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_2_{type}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_2_{type}.png",
#
#         f"{ic}/{band}/St_iC_CL_0_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_2_{type}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_1_{type}.png",
# ]
#
# montar_grid_topografia(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{type}_grid.png",
#     crop_list=[0, 300, 2100, 2000], rows=2, row_labels=["St kCar CL", "St iCar CL"]
# )
#
# # === 1220 cm-1 band - raw ===
# band = '1220'
# type = f'band_{band}_raw_global'
#
# kc = f"./figures/maps/St kC CLs/bands_40to1785_w-bg_nearest"
# ic = f"./figures/maps/St iC CLs/bands_40to1785_w-bg_nearest"
#
# paths_band = [
#         f"{kc}/{band}/St_kC_CL_0_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_7_Region_1_{type}.png",
#         f"{kc}/{band}/St_kC_CL_14_Region_2_{type}.png",
#         f"{kc}/{band}/St_kC_CL_21_Region_2_{type}.png",
#
#         f"{ic}/{band}/St_iC_CL_0_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_7_Region_2_{type}.png",
#         f"{ic}/{band}/St_iC_CL_14_Region_1_{type}.png",
#         f"{ic}/{band}/St_iC_CL_21_Region_1_{type}.png",
# ]
#
# montar_grid_topografia(
#     title=f"Band map at {band} 1/cm", image_paths=paths_band, save_path=f"{type}_grid.png",
#     crop_list=[0, 300, 2100, 2000], rows=2, row_labels=["St kCar CL", "St iCar CL"]
# )

# === 850 and 805 cm-1 band ===
band = '850'
type = f'band_{band}_diff_global'

kc = f"./figures/maps/St kC CLs/bands_40to1785_w-bg_nearest"
paths_band_kc = [
        f"{kc}/{band}/St_kC_CL_0_Region_1_{type}.png",
        f"{kc}/{band}/St_kC_CL_7_Region_1_{type}.png",
        f"{kc}/{band}/St_kC_CL_14_Region_2_{type}.png",
        f"{kc}/{band}/St_kC_CL_21_Region_2_{type}.png",]

band = '805'
type = f'band_{band}_diff_global'

ic = f"./figures/maps/St iC CLs/bands_40to1785_w-bg_nearest"
paths_band_ic = [
        f"{ic}/{band}/St_iC_CL_0_Region_1_{type}.png",
        f"{ic}/{band}/St_iC_CL_7_Region_2_{type}.png",
        f"{ic}/{band}/St_iC_CL_14_Region_1_{type}.png",
        f"{ic}/{band}/St_iC_CL_21_Region_1_{type}.png",]

paths_band = paths_band_kc + paths_band_ic

montar_grid_topografia(
    title=f"Band map at 850 1/cm (for kCar) and 805 (for iCar) 1/cm", image_paths=paths_band, save_path=f"{type}_grid.png",
    crop_list=[0, 300, 2100, 2000], rows=2, row_labels=["St kCar CL", "St iCar CL"]
)

# === 850 and 805 cm-1 band - raw ===
band = '850'
type = f'band_{band}_raw_global'

kc = f"./figures/maps/St kC CLs/bands_40to1785_w-bg_nearest"
paths_band_kc = [
        f"{kc}/{band}/St_kC_CL_0_Region_1_{type}.png",
        f"{kc}/{band}/St_kC_CL_7_Region_1_{type}.png",
        f"{kc}/{band}/St_kC_CL_14_Region_2_{type}.png",
        f"{kc}/{band}/St_kC_CL_21_Region_2_{type}.png",]

band = '805'
type = f'band_{band}_raw_global'

ic = f"./figures/maps/St iC CLs/bands_40to1785_w-bg_nearest"
paths_band_ic = [
        f"{ic}/{band}/St_iC_CL_0_Region_1_{type}.png",
        f"{ic}/{band}/St_iC_CL_7_Region_2_{type}.png",
        f"{ic}/{band}/St_iC_CL_14_Region_1_{type}.png",
        f"{ic}/{band}/St_iC_CL_21_Region_1_{type}.png",]

paths_band = paths_band_kc + paths_band_ic

montar_grid_topografia(
    title=f"Band map at 850 1/cm (for kCar) and 805 (for iCar) 1/cm", image_paths=paths_band, save_path=f"{type}_grid.png",
    crop_list=[0, 300, 2100, 2000], rows=2, row_labels=["St kCar CL", "St iCar CL"]
)
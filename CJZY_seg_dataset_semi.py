from osgeo import ogr, osr, gdal
import shapefile as shp
import os
import numpy as np
import cv2
import shutil
import pandas as pd
import Data_Processing.SHP as shp_m
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import geopandas

def label_png2tif(png_doc,save_doc,flag_tif_doc):
    id_list = []
    for file in os.listdir(r"K:\城中村数据\CJZY_uv"):
        if os.path.splitext(file)[1] == ".tif":
            name = os.path.splitext(file)[0]
            idx = name.split("_")[0]
            if idx not in id_list:
                id_list.append(idx)
    for id in tqdm(id_list):
        png_path = f"{png_doc}\\{id}_label_pred_oneband.png"
        flag_tif_path = f"{flag_tif_doc}\\{id}.tif"
        save_tif_doc = os.path.join(save_doc,"uv_tif")
        if not os.path.exists(save_tif_doc):
            os.mkdir(save_tif_doc)
        save_path = f"{save_tif_doc}\\{id}.tif"
        shp_m.png2tif(png_path,flag_tif_path,save_path)

def laebl_tif2shp(save_doc):
    tif_doc = os.path.join(save_doc,"uv_tif")
    save_shp_doc = os.path.join(save_doc, "uv_shp")
    if not os.path.exists(save_shp_doc):
        os.mkdir(save_shp_doc)
    for file in os.listdir(tif_doc):
        if os.path.splitext(file)[1] == ".tif":
            tif_path = os.path.join(tif_doc,file)
            shp_m.raster2shp(tif_path,save_shp_doc)


def merge_label_shp(save_shp,shp_doc):
    output_shp = shp.Writer(save_shp)
    shp_list = []
    flag = 0
    for shp_f in os.listdir(shp_doc):
        if os.path.splitext(shp_f)[1]==".shp":
            shp_list.append(os.path.join(shp_doc,shp_f))
    id = 1
    for shp_f in tqdm(shp_list):
        r = shp.Reader(shp_f)
        if flag==0:
            for field in r.fields[1:]:
                output_shp.field(*field)
            output_shp.field("uv_id","N",10)
            flag = 1
        for shapeRec in r.iterShapeRecords():
            value = shapeRec.record["value"]
            if value == 1:
                output_shp.record(*shapeRec.record,id)
                output_shp.shape(shapeRec.shape)
                id += 1
            else:
                continue

    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)  # 4326-GCS_WGS_1984; 4490- GCS_China_Geodetic_Coordinate_System_2000
    wkt = proj.ExportToWkt()
    coding = "UTF-8"

    f = open(save_shp.replace(".shp", ".prj"), 'w')
    g = open(save_shp.replace(".shp", ".cpg"), 'w')
    g.write(coding)
    f.write(wkt)
    g.close()
    f.close()


def area(shpPath):
    '''计算面积'''
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shpPath, 1)
    layer = dataSource.GetLayer()

    src_srs = layer.GetSpatialRef()  # 获取原始坐标系或投影
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(32649)  # WGS_1984_UTM_Zone_49N投影的ESPG号
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)  # 计算投影转换参数
    # geosr.SetWellKnownGeogCS("WGS_1984_UTM_Zone_49N")

    new_field = ogr.FieldDefn("Area", ogr.OFTReal)  # 创建新的字段
    new_field.SetWidth(32)
    new_field.SetPrecision(16)
    layer.CreateField(new_field)
    for feature in tqdm(layer):
        geom = feature.GetGeometryRef()
        geom2 = geom.Clone()
        geom2.Transform(transform)
        area_in_sq_m = geom2.GetArea()
        feature.SetField("Area", area_in_sq_m)
        layer.SetFeature(feature)
    dataSource.Destroy()


def delete_by_area(shp_path,save_shp):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    options = ["ENCODING=UTF-8"]

    r = driver.Open(shp_path, 0)
    layer = r.GetLayer()
    outputfilename = save_shp
    if os.path.exists(outputfilename):
        driver.DeleteDataSource(outputfilename)

    newds = driver.CreateDataSource(outputfilename)
    new_layer = newds.CopyLayer(layer, 'uv',options = options)


    for feature in tqdm(new_layer):
        area = feature.GetField("Area")
        if area<5000:
            fid = feature.GetFID()
            new_layer.DeleteFeature(int(fid))

    strSQL = "REPACK " + str(new_layer.GetName())
    newds.ExecuteSQL(strSQL, None, "")

    newds.Destroy()
    r.Destroy()



def png2shp_merge(png_doc,save_doc,flag_tif_doc,merge_shp_path,del_area_shp):
    label_png2tif(png_doc,save_doc,flag_tif_doc)
    laebl_tif2shp(save_doc)
    shp_doc = os.path.join(save_doc,"uv_shp")
    merge_label_shp(merge_shp_path,shp_doc)
    area(merge_shp_path)
    delete_by_area(merge_shp_path,del_area_shp)


def cut_grid_by_id(shp_path,save_doc,field_name):
    if not os.path.exists(save_doc):
        os.mkdir(save_doc)

    r = shp.Reader(shp_path)
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)  # 4326-GCS_WGS_1984; 4490- GCS_China_Geodetic_Coordinate_System_2000
    wkt = proj.ExportToWkt()
    coding = "UTF-8"

    for shapeRec in r.iterShapeRecords():
        rec = shapeRec.record[field_name]
        name =  str(rec)

        shp_name = os.path.join(save_doc,name)+".shp"
        w = shp.Writer(shp_name)
        for field in r.fields[1:]:
            w.field(*field)

        w.record(*shapeRec.record)
        w.shape(shapeRec.shape)
        w.close()

        f = open(shp_name.replace(".shp", ".prj"), 'w')
        g = open(shp_name.replace(".shp", ".cpg"), 'w')
        g.write(coding)
        f.write(wkt)
        g.close()
        f.close()


def image_clip():
    e_list = []
    for file in os.listdir(r"K:\城中村数据集\CJZY城中村置信度0.5-0.8数据集_20231107\image"):
        if os.path.splitext(file)[1] == ".tif":
            idx = file.split(".")[0]
            e_list.append(int(idx))

    for i in ["05", "06", "07", "08"]:
        r = shp.Reader(f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8样本\\{i}\\grid\\grid_city.shp")
        for rec in r.iterRecords():
            idx = rec["idx"]
            if idx not in e_list:
                e_list.append(idx)
                city = rec["Name"]
                clip_shp = f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8样本\\{i}\\grid\\split\\{idx}.shp"
                image_path = f"I:\\长江中游城市群谷歌高清影像\\{city}\\{city}_大图\\L18\\{city}.tif"
                save_path = f"K:\\城中村数据集\CJZY城中村置信度0.5-0.8数据集_20231107\\image\\{idx}.tif"
                shp_m.clip_Raster_shp(image_path, clip_shp, save_path,0)


def get_seg_label():
    for doc in ["05", "06", "07", "08"]:
        id_list = []
        for file in os.listdir(f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8样本\\{doc}\\uv_label_raster"):
            if os.path.splitext(file)[1] == ".tif":
                id_list.append(file.split(".")[0])

        for id in tqdm(id_list):
            label_path = f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8样本\\{doc}\\uv_label_raster\\{id}.tif"
            aoi_path = f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8样本\\aoi_500m_raster\\{id}.tif"
            save_path = f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8数据集_20231107\\label_{doc}\\{id}.tif"

            if os.path.exists(save_path):
                continue

            image_path = f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8数据集_20231107\\image\\{id}.tif"
            image_tif = gdal.Open(image_path)
            image_arr = image_tif.ReadAsArray()
            _,x_o, y_o = image_arr.shape

            label_tif = gdal.Open(label_path)
            label_arr = label_tif.ReadAsArray()
            x1,y1 = label_arr.shape

            aoi_tif = gdal.Open(aoi_path)
            band = aoi_tif.GetRasterBand(1)
            nodata = band.GetNoDataValue()
            aoi_arr = aoi_tif.ReadAsArray()
            x2,y2 = aoi_arr.shape

            save_arr = np.zeros((x_o,y_o),dtype="uint")

            x = min([x1, x2,  x_o])
            y = min([y1, y2,  y_o])

            label_arr_t = label_arr[:x,:y]
            aoi_arr_t = aoi_arr[:x,:y]
            aoi_arr_t[aoi_arr_t != nodata] = 2
            aoi_arr_t[aoi_arr_t == nodata] = 0
            res = np.sum(aoi_arr_t[label_arr_t == 1])/2
            uv_num = np.sum(label_arr_t)
            if uv_num == 0:
                pass
            else:
                coverage = res/uv_num

                if coverage>0.5:
                    print(id)
                    continue

            for i in range(x):
                for j in range(y):
                    if aoi_arr_t[i][j] == 2:
                        save_arr[i][j] = 2
                    elif label_arr_t[i][j] == 1:
                        save_arr[i][j] = 1
                    else:
                        pass
            proj =image_tif.GetProjection()
            gt = image_tif.GetGeoTransform()
            shp_m.save_tif_2(save_arr, save_path, y_o, x_o, 1, 1, gt, proj,0)



if __name__ == '__main__':
    # for i in ["5","6","7","8"]:
    #     png_path = f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8样本\\CJZY-UV-Confident-0.{i}-semiformer-0.5-out"
    #     save_doc = f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8样本\\0{i}"
    #     merge_shp_path = f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8样本\\0{i}\\uv.shp"
    #     del_area_shp = f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8样本\\0{i}\\uv_del_area.shp"
    #     png2shp_merge(png_path,save_doc,r"K:\城中村数据\CJZY_uv",merge_shp_path,del_area_shp)

    # for i in ["05","06","07","08"]:
    #     shp_path = f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8样本\\{i}\\grid\\grid_city.shp"
    #     save_doc = f"K:\\城中村数据集\\CJZY城中村置信度0.5-0.8样本\\{i}\\grid\\split"
    #     cut_grid_by_id(shp_path,save_doc,"idx")
    get_seg_label()
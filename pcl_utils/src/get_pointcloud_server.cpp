#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include "geometry_msgs/PoseStamped.h"
#include <typeinfo>

#include "pcl_ros/filters/filter.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>

#include "pcl_utils/snapshot.h"

#include <iostream>
#include <fstream>
#include <algorithm>

//boost
#include <boost/make_shared.hpp>

typedef boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> CloudPointRGBPtr;

struct ICPResult
{
  float icp_error;
  CloudPointRGBPtr cloud;
  Eigen::Matrix4d tf_matrix;
  float id;
};

std::vector<ICPResult> icp_result;

bool compare_highest_points(const ICPResult &result_1, const ICPResult &result_2) //minimum z is the higheset!!!!
{ 
    return result_1.icp_error < result_2.icp_error;
}

using namespace std;

std::string open_path_image = "../Solomon_TransparentObject_ws/src/pcl_utils/data/save_img/test_img.jpg";   // [1] img_file
std::string save_path_rgb = "../Solomon_TransparentObject_ws/src/pcl_utils/data/save_img/rgb_";             // [2] img_rs_rgb; [6] all_rs
std::string save_path_depth = "../Solomon_TransparentObject_ws/src/pcl_utils/data/save_img/depth_";         // [3] img_rs_depth; [6] all_rs
std::string open_path_cloud = "../Solomon_TransparentObject_ws/src/pcl_utils/data/save_cloud/test_cloud.pcd";// [4] cloud_file
std::string save_path_cloud = "../Solomon_TransparentObject_ws/src/pcl_utils/data/save_cloud/cloud";       // [5] cloud_rs; [6] all_rs

ros::Publisher pubFilter, pubColorRegion, pubSACSegmentation, pubExtractRegionSAC, pubICPAlign;

//===================Data===================
sensor_msgs::PointCloud2 Filter_output;   //宣告的輸出的點雲的格式

sensor_msgs::PointCloud2 colorbasedregiongrowingsegmentation_output;   //宣告的輸出的點雲的格式
sensor_msgs::PointCloud2 colorclass_output;   //宣告的輸出的點雲的格式
sensor_msgs::PointCloud2 SACsegmentation_output;   //宣告的輸出的點雲的格式
sensor_msgs::PointCloud2 Extract_Region_SAC_output;   //宣告的輸出的點雲的格式
sensor_msgs::PointCloud2 ICP_align_output;   //宣告的輸出的點雲的格式

//Origin Pointcloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
//filter Pointcloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
//PassThrough PointCloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr passthrough_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
//VoxelGrid PointCloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxelgrid_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
//RGB Class PointCloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr Class_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
//SAC
pcl::PointCloud<pcl::PointXYZRGB>::Ptr SACSegmentation_Extract_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
//color filter after Extract RegionGrowing with SACSegmentation
pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
//Extract RegionGrowing with SACSegmentation
pcl::PointCloud<pcl::PointXYZRGB>::Ptr Extract_RegionGrowing_SACSegmentation_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
//Result after do ICP
pcl::PointCloud<pcl::PointXYZRGB>::Ptr ICP_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > Extract_vector;
std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > ICP_filtered_vector;
std::vector <float> icp_score;




//===================Data===================/

//===================Config===================
float x_coordinate_min,y_coordinate_min,z_coordinate_min;
float x_coordinate_max,y_coordinate_max,z_coordinate_max;

int cnt_rgb = 0;
int cnt_depth = 0;
int cnt_cloud = 0;
int cnt_colorregion = 0;
int cnt_SACSegmentation = 0;
int cnt_ColorFiltered = 0;
int cnt_ExtractColor = 0;
//===================Config===================/

void print4x4Matrix (const Eigen::Matrix4d & matrix)
{
  cout<<matrix<<endl;
}

void set_coordinate_limit_min(const geometry_msgs::Point& input)
{
  x_coordinate_min = input.x;
  y_coordinate_min = input.y;
  z_coordinate_min = input.z;
  printf("coordinatee min   x:%f, y:%f, z:%f\n", x_coordinate_min, y_coordinate_min, z_coordinate_min);
}

void set_coordinate_limit_max(const geometry_msgs::Point& input)
{
  x_coordinate_max = input.x;
  y_coordinate_max = input.y;
  z_coordinate_max = input.z;
  printf("coordinatee max   x:%f, y:%f, z:%f\n", x_coordinate_max, y_coordinate_max, z_coordinate_max);
}

void FilterCloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  // //=================VoxelGrid=================
  // // 進行一個濾波處理
  // pcl::VoxelGrid<pcl::PointXYZRGB> sor;   //例項化濾波
  // sor.setInputCloud (cloud);     //設定輸入的濾波
  // sor.setLeafSize (0.1, 0.1, 0.1);   //設定體素網格的大小
  // sor.filter (*voxelgrid_filtered);      //儲存濾波後的點雲

  // // 再將濾波後的點雲的資料格式轉換為ROS 下的資料格式釋出出去
  // pcl::toROSMsg(*voxelgrid_filtered, voxelgridoutput);    //第一個引數是輸入，後面的是輸出
  // //釋出命令
  // pubVoxelGrid.publish (voxelgridoutput);
  // //=================VoxelGrid=================/

  // //=================transformation=================
  // /* Reminder: how transformation matrices work :

  //          |-------> This column is the translation
  //   | 1 0 0 x |  \
  //   | 0 1 0 y |   }-> The identity 3x3 matrix (no rotation) on the left
  //   | 0 0 1 z |  /
  //   | 0 0 0 1 |    -> We do not use this line (and it has to stay 0,0,0,1)

  //   METHOD #1: Using a Matrix4f
  //   This is the "manual" method, perfect to understand but error prone !
  // */
  // Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();

  // // Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
  // float theta = M_PI/4; // The angle of rotation in radians
  // transform_1 (0,0) = std::cos (theta);
  // transform_1 (0,1) = -sin(theta);
  // transform_1 (1,0) = sin (theta);
  // transform_1 (1,1) = std::cos (theta);
  // //    (row, column)

  // // Define a translation of 2.5 meters on the x axis.
  // transform_1 (0,3) = 2.5;

  // // Print the transformation
  // // printf ("Method #1: using a Matrix4f\n");
  // // std::cout << transform_1 << std::endl;

  // /*  METHOD #2: Using a Affine3f
  //   This method is easier and less error prone
  // */
  // Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();

  // // Define a translation of 2.5 meters on the x axis.
  // transform_2.translation() << 2.5, 0.0, 0.0;

  // // The same rotation matrix as before; theta radians around Z axis
  // transform_2.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));

  // // Print the transformation
  // // printf ("\nMethod #2: using an Affine3f\n");
  // // std::cout << transform_2.matrix() << std::endl;

  // // Executing the transformation
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
  // // You can either apply transform_1 or transform_2; they are the same
  // pcl::transformPointCloud (*passthrough_filtered, *transformed_cloud, transform_2);

  // // PCL to ROS
  // pcl::toROSMsg(*transformed_cloud, transform_output);    //第一個引數是輸入，後面的是輸出
  // //釋出命令
  // pubTransform.publish (transform_output);
  // //=================transformation=================/
}

bool savePointCloud(pcl_utils::snapshot::Request &req, pcl_utils::snapshot::Response &res)
{
  // req.call;
  // res.back;
  ostringstream os;
  os << cnt_cloud;
  string file_path_cloud = save_path_cloud + os.str();
  file_path_cloud = file_path_cloud + ".pcd";

  pcl::io::savePCDFileASCII<pcl::PointXYZRGB>(file_path_cloud, *cloud);
  
  cnt_cloud++;

  return true;
}

void do_savePCDFileASCII(std::string cloud_FileName, 
                         int *cnt, 
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud)
{
  ostringstream os;
  os << *cnt;
  string file_path_cloud = save_path_cloud + "_" + cloud_FileName + "_" + os.str();
  file_path_cloud = file_path_cloud + ".pcd";

  pcl::io::savePCDFileASCII<pcl::PointXYZRGB>(file_path_cloud, *input_cloud);
  *cnt = *cnt + 1;
}

void do_Passthrough(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, 
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud,
                    std::string dim, float min, float max)
{
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud (input_cloud);
  pass.setFilterFieldName (dim);
  pass.setFilterLimits (min, max);
    //pass.setFilterLimitsNegative (true);
  pass.filter (*output_cloud);
}

void do_VoxelGrid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, 
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud)
{
  // 進行一個濾波處理
  pcl::VoxelGrid<pcl::PointXYZRGB> sor;   //例項化濾波
  sor.setInputCloud (input_cloud);     //設定輸入的濾波
  sor.setLeafSize (0.005, 0.005, 0.005);   //設定體素網格的大小
  sor.filter (*output_cloud);      //儲存濾波後的點雲
}

void PointCloud_cd(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  // ROS to PCL
  pcl::fromROSMsg(*cloud_msg, *cloud);

  do_Passthrough(cloud, filter_cloud, "x", x_coordinate_min, x_coordinate_max);
  do_Passthrough(filter_cloud, filter_cloud, "y", y_coordinate_min, y_coordinate_max);
  do_Passthrough(filter_cloud, filter_cloud, "z", z_coordinate_min, z_coordinate_max);
  do_VoxelGrid(filter_cloud, filter_cloud);
  
  // PCL to ROS
  pcl::toROSMsg(*filter_cloud, Filter_output);    //第一個引數是輸入，後面的是輸出
  //Specify the frame that you want to publish
  Filter_output.header.frame_id = "camera_depth_optical_frame";
  //釋出命令
  pubFilter.publish (Filter_output);
}

void do_SACSegmentation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, 
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud)
{
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);//origin : 0.01
  seg.setInputCloud (input_cloud);
  seg.segment (*inliers, *coefficients);
  std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
  std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3] << std::endl;
  // 提取地面以外點雲
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  extract.setInputCloud (input_cloud);
  extract.setNegative (true);
  extract.setIndices (inliers);
  extract.filter (*output_cloud);

  // std::cerr << "Ground cloud after filtering: " << std::endl;
  // std::cerr << *output_cloud << std::endl;

 // PCL to ROS
  pcl::toROSMsg(*output_cloud, SACsegmentation_output);    //第一個引數是輸入，後面的是輸出

  //Specify the frame that you want to publish
  SACsegmentation_output.header.frame_id = "camera_depth_optical_frame";

  //釋出命令
  pubSACSegmentation.publish (SACsegmentation_output);
}

std::vector <pcl::PointIndices> 
do_RegionGrowingRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, 
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud)
{
  pcl::search::Search <pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);

  pcl::IndicesPtr indices (new std::vector <int>);

  pcl::PassThrough<pcl::PointXYZRGB> pass_indices;
  pass_indices.setInputCloud (input_cloud);
  pass_indices.setFilterFieldName ("z");
  pass_indices.setFilterLimits (-1*(std::numeric_limits<float>::max()), std::numeric_limits<float>::max());
  pass_indices.filter (*indices);

  pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
  reg.setInputCloud (input_cloud);
  reg.setIndices (indices);
  reg.setSearchMethod (tree);
  // Here the distance threshold is set. It is used to determine whether the point is neighbouring or not. 
  // If the point is located at a distance less than the given threshold, 
  // then it is considered to be neighbouring. 
  // It is used for clusters neighbours search.
  reg.setDistanceThreshold (10); 
  // This line sets the color threshold. 
  // Just as angle threshold is used for testing points normals in pcl::RegionGrowing to determine 
  // if the point belongs to cluster, this value is used for testing points colors.
  reg.setPointColorThreshold (3); 
  // Here the color threshold for clusters is set. 
  // This value is similar to the previous, but is used when the merging process takes place.
  reg.setRegionColorThreshold (5);
  // This value is similar to that which was used in the Region growing segmentation tutorial. 
  // In addition to that, it is used for merging process mentioned in the beginning. 
  // If cluster has less points than was set through setMinClusterSize method, 
  // then it will be merged with the nearest neighbour.
  reg.setMinClusterSize (2400);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);

  output_cloud = reg.getColoredCloud ();

  // PCL to ROS
  pcl::toROSMsg(*output_cloud, colorbasedregiongrowingsegmentation_output);    //第一個引數是輸入，後面的是輸出
  //Specify the frame that you want to publish
  colorbasedregiongrowingsegmentation_output.header.frame_id = "camera_depth_optical_frame";
  //釋出命令
  pubColorRegion.publish (colorbasedregiongrowingsegmentation_output);

  // do_Extract(output_cloud, Extract_RegionGrowing_SACSegmentation_filtered, clusters);
  return clusters;
}

Eigen::Vector4f do_ComputeLocation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud)
{
  Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*input_cloud, pcaCentroid);
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*input_cloud, pcaCentroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
	eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1)); //校正主方向间垂直
	eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
	eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0));
 
	std::cout << "特徵值va(3x1):\n" << eigenValuesPCA << std::endl;
	std::cout << "特徵向量ve(3x3):\n" << eigenVectorsPCA << std::endl;
	std::cout << "質心点(4x1):\n" << pcaCentroid << std::endl;
  return pcaCentroid;
}

void do_Extract(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, 
                std::vector <pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &output_cloud,
                std::vector <pcl::PointIndices> &clusters)
{

  output_cloud.clear();

  for(int kk = 0; kk < clusters.size(); kk++)
  {
      if(clusters[kk].indices.size() > 2000 && clusters[kk].indices.size() < 12000)
      {
          cout << "Cluster #" << kk << " size = " << clusters[kk].indices.size() << endl;
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
          tmp_cloud->clear();
          int counter = 0;

          while (counter < clusters[kk].indices.size ())
          {
            int idx = clusters[kk].indices[counter];
            tmp_cloud->push_back(input_cloud->points[idx]);
            counter++;        
          }
          output_cloud.push_back(tmp_cloud);
      }
  }

  // for (int i = 0; i < output_cloud.size() ; i++)
  // {
  //     // std::cout<< i <<" Extract_group "<<*output_cloud[i]<<endl;
  //       // PCL to ROS
  //     pcl::toROSMsg(*output_cloud[i], Extract_Region_SAC_output);    //第一個引數是輸入，後面的是輸出
  //     Extract_Region_SAC_output.header.frame_id = "camera_depth_optical_frame";
  //     pubExtractRegionSAC.publish (Extract_Region_SAC_output);
  // }  
  // for(int i = 0; i < output_cloud.size(); i++)
  // {
  //   do_ComputeLocation(output_cloud[i]);
  // }

  std::cout<<"do_Extract END!!"<<endl;
}

std::vector<Eigen::Matrix4d> 
do_ICP(std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > &input_cloud, 
       std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > &output_cloud,
       std::string source_data)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source(new pcl::PointCloud<pcl::PointXYZRGB>);

  std::vector<Eigen::Matrix4d>  final_transform;

  std::vector<int> ind;

  // std::string source_file = "/home/lucaubuntu18/Solomon_TransparentObject_ws/src/pcl_utils/data/cad_data/cloud_3_Extract_ColorRegion_SACSeg.pcd";
  // cout<<"source_file :" << source_data << endl;

  pcl::io::loadPCDFile<pcl::PointXYZRGB>(source_data, *source);
   
  pcl::removeNaNFromPointCloud(*source, *source, ind);
  output_cloud.clear();
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  // final_output_cloud->clear();
  // icp_score.clear();
  icp_result.clear();

  ICPResult  tmp_result;
  for (int i = 0; i < input_cloud.size(); i++)
  {
    //ICP
    int iterations = 50000;
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp; //创建ICP对象，用于ICP配准

    icp.setInputSource(source); //设置输入点云
    icp.setInputTarget(input_cloud[i]); //设置目标点云（输入点云进行仿射变换，得到目标点云）
    icp.setMaxCorrespondenceDistance(10);// Set the max correspondence distance (e.g., correspondences with higher distances will be ignored)
    icp.setMaximumIterations(iterations);    //criterion1: 设置最大迭代次数iterations=true
    icp.setTransformationEpsilon(1e-8);     //criterion2: transformation epsilon 
    icp.setEuclideanFitnessEpsilon(0.001);   // Set the euclidean distance difference epsilon (criterion 3)    

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr init_output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    init_output_cloud->clear();

    icp.align(*init_output_cloud);          //匹配后源点云
    
    output_cloud.push_back(init_output_cloud);

    //icp.setMaximumIterations(1);  // 设置为1以便下次调用
    std::cout << "Applied " << iterations << " ICP iteration(s)"<< std::endl;
    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

    // output_cloud->clear();
    if (icp.hasConverged())//icp.hasConverged ()=1（true）输出变换矩阵的适合性评估
    {
        std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
        // std::cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << std::endl;
        transformation_matrix = icp.getFinalTransformation().cast<double>();
        // print4x4Matrix(transformation_matrix);
        final_transform.push_back(transformation_matrix);
        // icp_score.push_back(icp.getFitnessScore());
        tmp_result.icp_error = icp.getFitnessScore();
        tmp_result.cloud = output_cloud[i];
        tmp_result.tf_matrix = transformation_matrix;
        icp_result.push_back(tmp_result);
        pcl::transformPointCloud(*source, *output_cloud[i], transformation_matrix);
    }
    else
    {
        PCL_ERROR("\nICP has not converged.\n");
    }
  // *final_output_cloud = *final_output_cloud + *output_cloud[i];
  }
  //   // PCL to ROS
  // pcl::toROSMsg(*final_output_cloud, ICP_align_output);    //第一個引數是輸入，後面的是輸出
  // //Specify the frame that you want to publish
  // ICP_align_output.header.frame_id = "camera_depth_optical_frame";
  // //釋出命令
  // pubICPAlign.publish (ICP_align_output);

  return final_transform;
}

void get_RegionYOLO(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, 
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud)
{
  //===========do some transform here===========
  // x_coordinate_min=;
  // x_coordinate_max=;
  // y_coordinate_min=;
  // y_coordinate_max=;
  // z_coordinate_min=;
  // z_coordinate_max=;
  //===========do some transform here===========/

  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud (input_cloud);
  pass.setFilterFieldName ("x");
  pass.setFilterLimits (x_coordinate_min, x_coordinate_max);
  pass.setFilterFieldName ("y");
  pass.setFilterLimits (y_coordinate_min, y_coordinate_max);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (z_coordinate_min, z_coordinate_max);
    //pass.setFilterLimitsNegative (true);
  pass.filter (*output_cloud);
}

// void file_formated()
// {
//   pcl::PointCloud<pcl::PointNormal>::Ptr source_normal(new pcl::PointCloud<pcl::PointNormal>);

//   std::string source_normal_file = "/home/lucaubuntu18/Solomon_TransparentObject_ws/src/open3d_utils/PCD_Data/cad_data/solomon_part_12000.pcd";
//   cout<<"source_file :" << source_normal_file << endl;

//   string file_path_cloud = save_path_cloud + "_" + source_normal_file + "_formated";
//   file_path_cloud = file_path_cloud + ".pcd";
//   pcl::io::loadPCDFile<pcl::PointNormal>(source_normal_file, *source_normal);
//   pcl::io::savePCDFileASCII<pcl::PointXYZRGB>(file_path_cloud, *source_normal);
// }

void set_InputFile(std::string file_name, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud)
{
  // cout<<"source_file : " << file_name << endl;
  pcl::io::loadPCDFile<pcl::PointXYZRGB>(file_name, *output_cloud);
}

template <typename T> 
vector<size_t> argsort(const vector<T> &v) 
{ 
  // 建立下标数组
  vector<size_t> idx(v.size()); 
  iota(idx.begin(), idx.end(), 0); 
  // 调用sort函数，匿名函数自动捕获待排序数组
  sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];}); 
  return idx;
}

bool Align(pcl_utils::snapshot::Request &req, pcl_utils::snapshot::Response &res)
{
  // std::string metal_location = "/home/lucaubuntu18/Solomon_TransparentObject_ws/src/pcl_utils/data/metal_location.txt";
  // std::ifstream file(metal_location);
  // std::string source_name ;

  // if(file.is_open())
  // {
  //     std::string line;
  //     while (std::getline(file, line)) {
  //         // using printf() in all tests for consistency
  //         source_name = line.c_str();
  //     }
  //     file.close();
  // }
  // std::cout<<"source_name : "<<source_name<<endl;

  std::string source_name_1 = "/home/lucaubuntu18/Solomon_TransparentObject_ws/src/pcl_utils/data/source/final/1.pcd";
  std::string source_name_2 = "/home/lucaubuntu18/Solomon_TransparentObject_ws/src/pcl_utils/data/source/final/2.pcd";
  std::string source_name_3 = "/home/lucaubuntu18/Solomon_TransparentObject_ws/src/pcl_utils/data/source/final/3.pcd";

  //=====================================================================================
  std::string target_location = "/home/lucaubuntu18/Solomon_TransparentObject_ws/src/pcl_utils/data/target_location.txt";
  std::ifstream file_target(target_location);
  std::string target_name ;

  std::vector<ICPResult> final_output_cloud;
  ICPResult temp_final;
  final_output_cloud.clear();

  if(file_target.is_open())
  {
      std::string line_target;
      while (std::getline(file_target, line_target)) {
          // using printf() in all tests for consistency
          target_name = line_target.c_str();
      }
      file_target.close();
  }
  std::cout<<"target_name_name : "<<target_name<<endl;
  //=================Filters=================
  do_Passthrough(cloud, cloud, "x", x_coordinate_min, x_coordinate_max);
  do_Passthrough(cloud, cloud, "y", y_coordinate_min, y_coordinate_max);
  do_Passthrough(cloud, cloud, "z", z_coordinate_min, z_coordinate_max);
  // do_VoxelGrid(cloud, cloud);

  // set_InputFile(target_name, cloud);
  //=================Filters=================/

  //=================PlaneModelSegmentation and ExtractGround=================
  do_SACSegmentation(cloud, SACSegmentation_Extract_filtered);
  do_savePCDFileASCII("SAC_Extract", &cnt_SACSegmentation, SACSegmentation_Extract_filtered);
  //=================PlaneModelSegmentation=================/

  //=================RegionGrowing=================
  std::vector <pcl::PointIndices> clusters = do_RegionGrowingRGB(SACSegmentation_Extract_filtered, color_filtered);
  do_savePCDFileASCII("Color_Region", &cnt_ColorFiltered, color_filtered);
  //=================RegionGrowing=================/

  //=================Extract RegionGrowing=================
  do_Extract(color_filtered, Extract_vector, clusters);
  for (int i=0; i<Extract_vector.size(); i++ )
  {
    do_savePCDFileASCII("Extract_vector", &cnt_ExtractColor, Extract_vector[i]);
  }
  //=================Extract RegionGrowing=================/

  //=================ICP1=================
  std::vector<Eigen::Matrix4d> Metal_Transform1;
  Metal_Transform1 = do_ICP(Extract_vector, ICP_filtered_vector, source_name_1);

  std::cout<<"Final transform list 1: "<<endl;

  for (int i=0; i<Metal_Transform1.size(); i++)
  {
    std::cout<<"Metal Object["<<i<<"] :"<<endl;
    std::cout<<icp_result[i].icp_error<<endl;
    print4x4Matrix(Metal_Transform1[i]);
  }
  
  std::sort(icp_result.begin(), icp_result.end(), compare_highest_points);

  for(int i =0; i<icp_result.size();i++)
  {
    std::cout<<icp_result[i].icp_error<<endl;
  }
  //=================ICP1=================/
  if (icp_result.size()!= 0)
  {
    temp_final.icp_error = icp_result[0].icp_error;
    temp_final.cloud = icp_result[0].cloud;
    temp_final.tf_matrix = icp_result[0].tf_matrix;
    temp_final.id = 1;
    final_output_cloud.push_back(temp_final);
  }
  //=================ICP2=================
  std::vector<Eigen::Matrix4d> Metal_Transform2;
  Metal_Transform2 = do_ICP(Extract_vector, ICP_filtered_vector, source_name_2);

  std::cout<<"Final transform list 2: "<<endl;

  for (int i=0; i<Metal_Transform2.size(); i++)
  {
    std::cout<<"Metal Object["<<i<<"] :"<<endl;
    print4x4Matrix(Metal_Transform2[i]);
  }

  std::sort(icp_result.begin(), icp_result.end(), compare_highest_points);

  for(int i =0; i<icp_result.size();i++)
  {
    std::cout<<icp_result[i].icp_error<<endl;
  }
  //=================ICP2=================/
  if (icp_result.size()!= 0)
  {
    temp_final.icp_error = icp_result[0].icp_error;
    temp_final.cloud = icp_result[0].cloud;
    temp_final.tf_matrix = icp_result[0].tf_matrix;
    temp_final.id = 2;
    final_output_cloud.push_back(temp_final);
  }

  //=================ICP3=================
  std::vector<Eigen::Matrix4d> Metal_Transform3;
  Metal_Transform3 = do_ICP(Extract_vector, ICP_filtered_vector, source_name_3);

  std::cout<<"Final transform list 3: "<<endl;

  for (int i=0; i<Metal_Transform3.size(); i++)
  {
    std::cout<<"Metal Object["<<i<<"] :"<<endl;
    print4x4Matrix(Metal_Transform3[i]);
  }

  std::sort(icp_result.begin(), icp_result.end(), compare_highest_points);

  for(int i =0; i<icp_result.size();i++)
  {
    std::cout<<icp_result[i].icp_error<<endl;
  }
  //=================ICP3=================/
  if (icp_result.size()!= 0)
  {
    temp_final.icp_error = icp_result[0].icp_error;
    temp_final.cloud = icp_result[0].cloud;
    temp_final.tf_matrix = icp_result[0].tf_matrix;
    temp_final.id = 3;
    final_output_cloud.push_back(temp_final);
  }


  std::sort(final_output_cloud.begin(), final_output_cloud.end(), compare_highest_points);

  std::cout<<"final_output_cloud size "<<final_output_cloud.size()<<endl;

  for(int i=0; i<final_output_cloud.size(); i++)
  {
     std::cout<<final_output_cloud[i].icp_error<<endl;
  }
  Eigen::Vector4f final_location;
  final_location = do_ComputeLocation(final_output_cloud[0].cloud);
  std::cout << "質心点(4x1):\n" << final_location << std::endl;
  std::cout << "id:\n" << final_output_cloud[0].id << std::endl;

    // // PCL to ROS
  pcl::toROSMsg(*final_output_cloud[0].cloud, ICP_align_output);    //第一個引數是輸入，後面的是輸出
  //Specify the frame that you want to publish
  ICP_align_output.header.frame_id = "camera_depth_optical_frame";
  //釋出命令
  pubICPAlign.publish (ICP_align_output);

  return true;
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "get_pointcloud_server");

  ros::NodeHandle nh;

  // Create ROS preccess pointcloud publisher for the output point cloud
  pubFilter = nh.advertise<sensor_msgs::PointCloud2> ("/process_pubFilter", 1);
  pubColorRegion = nh.advertise<sensor_msgs::PointCloud2> ("/process_ColorRegion_cloud", 1);
  pubSACSegmentation = nh.advertise<sensor_msgs::PointCloud2> ("/process_SACSegmentation_cloud", 1);
  // pubExtractRegionSAC = nh.advertise<sensor_msgs::PointCloud2> ("/process_ExtractRegionSAC_cloud", 1);
  pubICPAlign = nh.advertise<sensor_msgs::PointCloud2> ("/process_puICPAlign_cloud", 1);

  // Create ROS config publisher for the passthrough limits
  ros::Publisher pubCoordinateLimitMin = nh.advertise<geometry_msgs::Point> ("/coordinate_limit_min", 1);
  ros::Publisher pubCoordinateLimitMax = nh.advertise<geometry_msgs::Point> ("/coordinate_limit_max", 1);

  // Create ROS subscriber for the input point cloud
  ros::Subscriber subSaveCloud = nh.subscribe<sensor_msgs::PointCloud2> ("/camera/depth/color/points", 1, PointCloud_cd);
  ros::Subscriber subCoordinateMin = nh.subscribe ("/coordinate_limit_min", 1, set_coordinate_limit_min);
  ros::Subscriber subCoordinateMax = nh.subscribe ("/coordinate_limit_max", 1, set_coordinate_limit_max);

  // Create a ROS ServiceServer for save pointcloud service 
  ros::ServiceServer savePointCloud_service = nh.advertiseService("snapshot", savePointCloud);
  ros::ServiceServer Align_service = nh.advertiseService("AlignPointCloud", Align);

  geometry_msgs::Point min, max;

  if (argc==1)
  {
    min.x = -1.0;
    min.y = -1.0;
    min.z = -1.0;

    max.x = 1.0;
    max.y = 0.0;
    max.z = 0.3;
  }
  else if(argc==7)
  {
    min.x = atof(argv[1]);
    max.x = atof(argv[2]);

    min.y = atof(argv[3]);
    max.y = atof(argv[4]);

    min.z = atof(argv[5]);
    max.z = atof(argv[6]);
  }
  else
  {
    min.x = -1.0;
    min.y = -1.0;
    min.z = -1.0;

    max.x = 1.0;
    max.y = 1.0;
    max.z = 0.3;
  }
  pubCoordinateLimitMin.publish(min);
  pubCoordinateLimitMax.publish(max);

  ros::spin();
}
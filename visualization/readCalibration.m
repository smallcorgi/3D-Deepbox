function P = readCalibration(calib_dir,img_idx,cam)

  file_list = dir(calib_dir);
  % load 3x4 projection matrix
  P = dlmread(sprintf('%s/%s',calib_dir,file_list(3+img_idx).name),' ',0,1);
  P = P(cam+1,:);
  P = reshape(P ,[4,3])';
  
end

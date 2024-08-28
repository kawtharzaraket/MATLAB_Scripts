clear, clc;
Array_of_features = {};
No_of_images = 3470;
%%3470%%1874%%2813
for k = 2814 : No_of_images
   image = ['Outlier_Dataset/Image(',num2str(k,'%d'),').jpg'];
    I = rgb2gray(imread(image));
    LBP = extractLBPFeatures(I);
    GLCM = reshape(graycomatrix(I),1,[]);
    HOG = extractHOGFeatures(I);
   Array_of_features = [Array_of_features;LBP;GLCM;HOG];
end
writetable(cell2table(Array_of_features),'ExtractedFeatures4.csv','WriteVariableNames',false,'QuoteStrings',true);
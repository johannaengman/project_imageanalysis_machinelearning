function Icrop = readim(folder,sizex,sizey)
    I = cell(length(folder)-2, 1);
    nbrim = 0;
    for i = 3:length(folder)
       I{i-2} = imresize(imread(fullfile(folder(i).folder, folder(i).name)),0.2);
       I{i-2} = rgb2gray(I{i-2}(:,:,1:3));
       nbrim = nbrim + floor(size(I{i-2},2)/sizex)*floor(size(I{i-2},1)/sizey);
    end
    
    Icrop = zeros(sizey,sizex,1,nbrim);
    tempnbrim = 0;
    for l=1:length(I)
        im = I{l};
        y = floor(size(im,1)/sizey);
        x = floor(size(im,2)/sizex);

        for i = 1:x
            for k = 1:y
                Icrop(:,:,1,i+x*(k-1)+tempnbrim) = im((k-1)*sizey+1:k*sizey, (i-1)*sizex+1:i*sizex,:);
            end 
        end
        tempnbrim = tempnbrim + x*y;
    end
end
clear;
clc;

%%
%Removes the folder containing interemediate images if present
if (exist('midimages','dir'))>0;
    rmdir('midimages','s');
end
%%
clc;
im1=0;
im2=0;
%%
if exist('warp_triangle_double.mexw64', 'file') ~=2
    cd('functions')
	mex('warp_triangle_double.c','image_interpolation.c');
	cd ..
else if exist('warp_triangle_double.mexw32', 'file') ~=2
        cd('functions')
		mex('warp_triangle_double.c','image_interpolation.c');
		cd ..
    end
end
if exist('getFaceFeatures.mexw64', 'file') ~=2
    mex getFaceFeatures.cpp;
else if exist('getFaceFeatures.mexw32', 'file') ~=2
        mex getFaceFeatures.cpp;
    end
end
%%

%Reads the file names from user
Image1='';
Image2='';
while true
    while true
        Image1 = input('Enter first image name ','s');
        if (exist(Image1,'file'))>0;
            im1 = imread(Image1);
            break;
        else
            disp('Given file does not exist');
        end
    end

    while true
        Image1 = input('Enter second image name ','s');
        if (exist(Image1,'file'))>0;
            im2 = imread(Image1);
            break;
        else
            disp('Given file does not exist');
        end
    end
    
    if size(size(im1),2)~=size(size(im2),2)
        disp('Both images should be GRAY or RGB');
    else
        clc;
        break;
    end
end
%%
flow=0;

input_points=0;
base_points=0;
%%
%Promts user to select control points or to generate intermediate images
while true
    disp('---------------------------------------');
    disp('Press 1 to select control points');
    disp('Press 2 to generate Intermediate Images');
    disp('Press 3 to exit');
    disp('---------------------------------------');
    
    inp=input('');
    if(inp==3)
        clear;
        clc;
        break;
    end
    if(inp>2 || inp<1)
        disp('<-----Please enter valid option----->');
        continue;
    end
    if(inp==1)
        while(1)
            clc;
            disp('<-----Enter 1 for manual selection----->');
            disp('<-----Enter 2 for automatic selection----->');
            inpui=input('');
            if(inpui==1)
                [input_points, base_points]=cpselect(im1,im2,'Wait',true);
                break;
            end
            if(inpui==2)
                input_points = GetFaceFeatures(Image1);
                base_points = GetFaceFeatures(Image2);
                break;
            end
        end
        clc;
    end
    if(inp==2 && flow==0)
        disp('<-----Please select control points first----->');
        continue;
    elseif inp==2
        nof=input('Enter number of frames : ');

        DoMorph(im1,im2,input_points,base_points,nof);
        clc;
    end
    
    flow=inp;
end
function DoMorph(im1,im2,input_points,base_points,nof)
    im1 = im2double(im1);
    im2 = im2double(im2);
    imsize=size(im1);
    H=imsize(1,1);
    W=imsize(1,2);

    append = zeros(4,2);
    append(1,:)=[1,1];
    append(2,:)=[round(W/2),1];
    append(3,:)=[W,1];
    append(4,:)=[1,round(H/2)];
    append(5,:)=[W,round(H/2)];
    append(6,:)=[1,H];
    append(7,:)=[round(W/2),H];
    append(8,:)=[W,H];

    i_points=[append;input_points];
    b_points=[append;base_points];

    Trimesh = delaunay(b_points);

    mkdir('midimages');   
    n=nof+1;

    for g=1:nof
        temp = (((n-g)*input_points) + (g*base_points))/n;
        temp=round(temp);
        mid_points = [append;temp];
        J1=warp_triangle(im1,i_points,mid_points,[H,W],Trimesh);
        J2=warp_triangle(im2,b_points,mid_points,[H,W],Trimesh);
        MidC = (((n-g)*J1)+(g*J2))/n;

        filename=sprintf('mid%d.jpg',g);
        cd midimages;
        imwrite(MidC,filename);
        cd ..;
    end
end
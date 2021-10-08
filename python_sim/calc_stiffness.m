function stiffness = calc_stiffness(pixel_matrix_path)
    tic
    
    %
    'FIRST'
    im_raw=imread(pixel_matrix_path); %%input pixel matrix
    imshow(im_raw)
    x = double(rgb2gray(im_raw));

    'SECOND'
    if ndims(x)==3
        x = x(:,:,1);
    end
    imshow(x)
    'THIRD'
    x(x<254) = 0;

    % trim empty edges:
    for i = 1:20
        if all(x(i,:) == 255)
            x(1,:) = [];
            'TRIM BOTTOM'
        end
        if all(x(:,i) == 255)
            x(:,1) = [];
            'TRIM LEFT'
        end
        if all(x(end,:) == 255)
            x(end,:) = [];
            'TRIM TOP'
        end
        if all(x(:,end) == 255)
            x(:,end) = [];
            'TRIM RIGHT'
        end
    end

    x = (255-x)/255;
    imshow(x)
    [nely, nelx] = size(x)
    E0 = 10;
    Emin = 1e-9;
    nu = 0.3;
    A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
    A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
    B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
    B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
    KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);
    nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
    edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
    edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
    iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
    jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);
    %loading condition

    F = sparse(2*(nely+1)*(nelx+1),1); %%Force=1  F/U=E_computated E_materials=10; E_normalized=E_computated/E_material
    for i = 1:nelx+1
        F(2*((i-1)*(nely+1)+1),1)=-1;
    end
    fixeddofs(1)=2*(nely+1)-1;
    fixeddofs(2)=2*(nely+1);
    for i =3:nelx+2
    fixeddofs(i)=fixeddofs(i-1)+(nely+1)*2;
    end

    alldofs     = [1:2*(nely+1)*(nelx+1)];
    freedofs    = setdiff(alldofs,fixeddofs);

    %%FE
    "DOING FEA"
        
    reshape_arg1 = KE(:)*(Emin+x(:)');
    reshape_arg2 = reshape_arg1.*(E0-Emin);
    sK = reshape(reshape_arg2,64*nelx*nely,1);
    K = sparse(iK,jK,sK); K = (K+K')/2;
    U(freedofs) = K(freedofs,freedofs)\F(freedofs);

    for j=1:nelx+1
        deformation(j)=U(2*((j-1)*(nely+1)+1));
    end

    Mi_deformation=mean(deformation);
    toc
    imshow(x)
    stiffness=[-nelx/Mi_deformation, sum(sum(x))/(nelx*nely)];

end
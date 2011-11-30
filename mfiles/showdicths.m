function flatDict = showdicths(dict, dimVec, numRows, numCols, varargin)

direction = 'columnmajor';
highcontrast = 0;
drawlines = 0;
linecolor = 0;

for i = 1:length(varargin)
	if (~ischar(varargin{i}))
		continue;
	end;
	switch(varargin{i})
	case 'highcontrast'
		highcontrast = 1;
	case 'lines'
		drawlines = 1;
	case 'whitelines'
		drawlines = 1;
		linecolor = 1;
	case 'rowmajor'
		direction = 'rowmajor';
	end;
end;

if (length(dimVec) == 3),
	flatDict = zeros(dimVec(1) * numRows, dimVec(2) * numCols, 3);
else
	flatDict = zeros(dimVec(1) * numRows, dimVec(2) * numCols, 1);
	dimVec(3) = 1;
end

numAtoms = size(dict, 2);
iterDict = 0;
if (strcmp(direction,'columnmajor')),
	for iterY = 1:numCols,
		for iterX = 1:numRows,
			iterDict = iterDict + 1;
			if (iterDict > numAtoms),
				break;
			end;
			flatDict((iterX - 1) * dimVec(1) + (1:dimVec(1)), (iterY - 1) * dimVec(2) + (1:dimVec(2)), :) = ...
				getrgb(reshape(dict(:, iterDict), dimVec));
			if (highcontrast == 1),
				flatDict((iterX - 1) * dimVec(1) + (1:dimVec(1)), (iterY - 1) * dimVec(2) + (1:dimVec(2)), :) = ...
					imnorm(flatDict((iterX - 1) * dimVec(1) + (1:dimVec(1)), (iterY - 1) * dimVec(2) + (1:dimVec(2)), :), [0 1]);
			end;
		end;
	end;
elseif (strcmp(direction,'rowmajor')),
	for iterX = 1:numRows,
		for iterY = 1:numCols,
			iterDict = iterDict + 1;
			if (iterDict > numAtoms),
				break;
			end;
			flatDict((iterX - 1) * dimVec(1) + (1:dimVec(1)), (iterY - 1) * dimVec(2) + (1:dimVec(2)), :) = ...
				getrgb(reshape(dict(:, iterDict), dimVec));
			if (highcontrast == 1),
				flatDict((iterX - 1) * dimVec(1) + (1:dimVec(1)), (iterY - 1) * dimVec(2) + (1:dimVec(2)), :) = ...
					imnorm(flatDict((iterX - 1) * dimVec(1) + (1:dimVec(1)), (iterY - 1) * dimVec(2) + (1:dimVec(2)), :), [0 1]);
			end;
		end;
	end;
end;

if (drawlines),
	for iterX = 1:numRows,
		if (linecolor == 0),
			flatDict(iterX * dimVec(1), :, :) = 0;
		else
			flatDict(iterX * dimVec(1), :, :) = 1;
		end;
	end;
	for iterY = 1:numCols,
		if (linecolor == 0),
			flatDict(:, iterY * dimVec(2), :) = 0;
		else
			flatDict(:, iterY * dimVec(2), :) = 1;
		end;
	end;
end;

imshow(imnorm(flatDict, [0 1]), []);

classdef SVMs
    properties
        class
        SVs
        w
        b
    end
    
    methods
        function obj = SVMs(class, SVs, w, b)
            obj.class = class;
            obj.SVs = SVs;
            obj.w = w;
            obj.b = b;
        end
    end
end

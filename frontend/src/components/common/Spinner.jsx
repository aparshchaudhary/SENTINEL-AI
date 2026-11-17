import React from 'react';
import { Loader2 } from 'lucide-react';

export const Spinner = ({ size = 'md', className = '', label }) => {
  const sizeMap = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-10 h-10',
    xl: 'w-14 h-14',
  };

  return (
    <div className={`flex flex-col items-center justify-center gap-3 ${className}`}>
      <Loader2 className={`animate-spin text-purple-500 ${sizeMap[size] || sizeMap.md}`} />
      {label && <p className="text-xs font-medium text-slate-400 tracking-wide">{label}</p>}
    </div>
  );
};

export default Spinner;

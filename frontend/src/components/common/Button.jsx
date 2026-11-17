import React from 'react';
import { Loader2 } from 'lucide-react';

const VARIANTS = {
  primary: 'bg-gradient-to-r from-brand-purple to-brand-blue text-white hover:opacity-95 shadow-md shadow-brand-purple/20 border border-transparent',
  secondary: 'bg-slate-800 text-slate-200 hover:bg-slate-700/80 border border-slate-700/80 shadow-sm dark:bg-slate-800 dark:hover:bg-slate-700/80 light:bg-slate-100 light:text-slate-800 light:border-slate-300 light:hover:bg-slate-200',
  success: 'bg-emerald-600 text-white hover:bg-emerald-500 shadow-sm shadow-emerald-600/20 border border-transparent',
  danger: 'bg-rose-600 text-white hover:bg-rose-500 shadow-sm shadow-rose-600/20 border border-transparent',
  ghost: 'bg-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent',
  outline: 'bg-transparent border border-brand-purple/50 text-purple-400 hover:bg-purple-950/30',
};

const SIZES = {
  sm: 'px-3 py-1.5 text-xs rounded-lg gap-1.5',
  md: 'px-4 py-2 text-sm rounded-xl gap-2',
  lg: 'px-6 py-2.5 text-base rounded-xl gap-2.5 font-semibold',
};

export const Button = ({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  icon: Icon,
  className = '',
  onClick,
  type = 'button',
  ...props
}) => {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      onClick={onClick}
      className={`inline-flex items-center justify-center font-medium transition-all duration-150 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100 ${VARIANTS[variant] || VARIANTS.primary} ${SIZES[size] || SIZES.md} ${className}`}
      {...props}
    >
      {loading ? (
        <Loader2 className="w-4 h-4 animate-spin text-current" />
      ) : Icon ? (
        <Icon className="w-4 h-4 shrink-0" />
      ) : null}
      {children}
    </button>
  );
};

export default Button;

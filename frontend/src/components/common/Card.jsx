import React from 'react';

export const Card = ({
  children,
  className = '',
  title,
  subtitle,
  headerAction,
  footer,
  noPadding = false,
  ...props
}) => {
  return (
    <div
      className={`bg-surface-dark dark:bg-slate-900/80 border border-slate-800/80 dark:border-slate-800/80 light:bg-white light:border-slate-200 rounded-2xl shadow-xl shadow-black/10 overflow-hidden transition-all duration-200 ${className}`}
      {...props}
    >
      {(title || subtitle || headerAction) && (
        <div className="px-6 py-4 border-b border-slate-800/80 light:border-slate-200 flex items-center justify-between gap-4">
          <div>
            {title && (
              <h3 className="text-base font-semibold text-slate-100 light:text-slate-900 tracking-tight">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-xs text-slate-400 light:text-slate-500 mt-0.5">
                {subtitle}
              </p>
            )}
          </div>
          {headerAction && <div className="shrink-0">{headerAction}</div>}
        </div>
      )}

      <div className={noPadding ? '' : 'p-6'}>{children}</div>

      {footer && (
        <div className="px-6 py-3.5 bg-slate-950/40 light:bg-slate-50 border-t border-slate-800/80 light:border-slate-200 text-xs text-slate-400 light:text-slate-500">
          {footer}
        </div>
      )}
    </div>
  );
};

export default Card;

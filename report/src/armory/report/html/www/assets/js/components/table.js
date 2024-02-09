export const Table = {
    template: `
        <table class="text-left text-sm w-full">
            <slot></slot>
        </table>
    `,
};

export const TableHead = {
    template: `
        <thead class="bg-zinc-200 text-xs uppercase">
            <slot></slot>
        </thead>
    `,
};

export const TableHeader = {
    template: `
        <th scope="col" class="px-6 py-3">
            <slot></slot>
        </th>
    `,
};

export const TableBody = {
    template: `<tbody><slot></slot></tbody>`,
};

export const TableRow = {
    template: `
        <tr class="border-b even:bg-zinc-50">
            <slot></slot>
        </tr>
    `,
};

export const TableRowHeader = {
    template: `
        <th scope="row" class="px-6 py-4 font-medium whitespace-nowrap">
            <slot></slot>
        </th>
    `,
};

export const TableCell = {
    template: `
        <td class="px-6 py-4">
            <slot></slot>
        </td>
    `,
};

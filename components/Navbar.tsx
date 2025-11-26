'use client'
import { Fragment } from 'react'
import { usePathname } from 'next/navigation'
import {
  Disclosure,
  DisclosureButton,
  DisclosurePanel,
  Menu,
  MenuButton,
  MenuItem,
  MenuItems,
  Transition,
} from '@headlessui/react'
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'

import GroupIcon from './icons/groupIcon'
import CalendarTodayIcon from './icons/CalendarIcon'
import TestLogo from './icons/TestLogo'
import SettingsIcon from './icons/SettingIcon'
import MenuIcon from './icons/MenuIcon'

const navigation = [
  { name: 'Patients', href: '/', svgIcon: <GroupIcon /> },
  { name: 'Readmission Prediction', href: '/readmission', svgIcon: <CalendarTodayIcon /> },
]

//@ts-ignore
function classNames(...classes) {
  return classes.filter(Boolean).join(' ')
}

const Navbar = () => {
  const pathname = usePathname() // 👈 get current URL path

  return (
    <Disclosure as="nav" className="bg-white mt-4 mb-8 mx-4 rounded-none lg:rounded-full">
      {({ open }) => (
        <>
          <div className="sm:px-6 lg:px-8">
            <div className="relative flex h-16 items-center justify-between">
              <div className="absolute inset-y-0 left-0 flex items-center lg:hidden">
                <DisclosureButton className="relative inline-flex items-center justify-center rounded-md p-2 text-gray-400 hover:bg-gray-700 hover:text-white focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white">
                  <span className="absolute -inset-0.5" />
                  <span className="sr-only">Open main menu</span>
                  {open ? (
                    <XMarkIcon className="block h-6 w-6" aria-hidden="true" />
                  ) : (
                    <Bars3Icon className="block h-6 w-6" aria-hidden="true" />
                  )}
                </DisclosureButton>
              </div>

              <div className="flex flex-1 md:items-center justify-center ">
                <div className="flex flex-shrink-0 items-center">
                  <TestLogo />
                </div>
                <div className="hidden lg:w-full sm:ml-6 sm:block ">
                  <div className="hidden lg:flex justify-center space-x-4">
                    {navigation.map((item) => (
                      <a
                        key={item.name}
                        href={item.href}
                        className={classNames(
                          pathname === item.href
                            ? 'bg-teal-300 p-4 text-sm font-medium text-black mx-4 hover:bg-teal-300 hover:text-black'
                            : 'bg-white p-4 text-sm font-medium text-black mx-4 hover:bg-teal-300 hover:text-black',
                          'flex rounded-full px-3 py-2 text-sm font-medium'
                        )}
                        aria-current={pathname === item.href ? 'page' : undefined}
                      >
                        <div className='pr-2'>{item.svgIcon}</div>
                        {item.name}
                      </a>
                    ))}
                  </div>
                </div>
              </div>

              <div className="absolute inset-y-0 right-0 md:flex items-center pr-2 sm:static sm:inset-auto hidden sm:pr-0">
                <Menu as="div" className="relative ml-3">
                  <div className='flex'>
                    <MenuButton className="relative flex rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-gray-800">
                      <span className="absolute -inset-1.5" />
                      <span className="sr-only">Open user menu</span>
                      <img
                        className="h-4 w-4 lg:h-8 lg:w-8 rounded-full"
                        src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                        alt=""
                      />
                    </MenuButton>
                    <div className='grid grid-cols-1 grid-row-1 mx-2'>
                      <p className='text-xs font-medium'>Dr Jose Simmons</p>
                      <p className='text-xs font-light'>General Practitioner</p>
                    </div>
                    <div className=' border-slate-300 border mr-1 border-dotted' ></div>
                    <button type="button" className="px-2">
                      <SettingsIcon className="h-5 w-5" />
                    </button>
                    <button type="button">
                      <MenuIcon className="h-5 w-5" />
                    </button>
                  </div>

                  <Transition
                    enter="transition ease-out duration-100"
                    enterFrom="transform opacity-0 scale-95"
                    enterTo="transform opacity-100 scale-100"
                    leave="transition ease-in duration-75"
                    leaveFrom="transform opacity-100 scale-100"
                    leaveTo="transform opacity-0 scale-95"
                  >
                    <MenuItems className="absolute right-0 z-10 mt-2 w-48 origin-top-right rounded-md bg-white py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                      {navigation.map((item, index) => (
                        <MenuItem key={index}>
                          {({ focus }) => (
                            <a
                              href={item.href}
                              className={classNames(focus ? 'bg-gray-100' : '', 'block px-4 py-2 text-sm text-gray-700')}
                            >
                              <div className='flex pr-2'>
                                <p className='mr-2'>{item.svgIcon}</p>
                                <p>{item.name}</p>
                              </div>
                            </a>
                          )}
                        </MenuItem>
                      ))}
                      <MenuItem>
                        {({ focus }) => (
                          <a
                            href="#"
                            className={classNames(focus ? 'bg-gray-100' : '', 'block px-4 py-2 text-sm text-gray-700')}
                          >
                            Settings
                          </a>
                        )}
                      </MenuItem>
                      <MenuItem>
                        {({ focus }) => (
                          <a
                            href="#"
                            className={classNames(focus ? 'bg-gray-100' : '', 'block px-4 py-2 text-sm text-gray-700')}
                          >
                            Sign out
                          </a>
                        )}
                      </MenuItem>
                    </MenuItems>
                  </Transition>
                </Menu>
              </div>
            </div>
          </div>

          <DisclosurePanel className="lg:hidden">
            <div className="space-y-1 px-2 pb-3 pt-2">
              {navigation.map((item) => (
                <DisclosureButton
                  key={item.name}
                  as="a"
                  href={item.href}
                  className={classNames(
                    pathname === item.href
                      ? 'bg-teal-300 p-4 text-sm font-medium text-black mx-4 hover:bg-teal-300 hover:text-black'
                      : 'bg-white p-4 text-sm font-medium text-black mx-4 hover:bg-teal-300 hover:text-black',
                    'flex rounded-full px-3 py-2 text-sm font-medium'
                  )}
                  aria-current={pathname === item.href ? 'page' : undefined}
                >
                  {item.name}
                </DisclosureButton>
              ))}
            </div>
          </DisclosurePanel>
        </>
      )}
    </Disclosure>
  )
}

export default Navbar
